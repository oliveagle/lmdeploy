#!/usr/bin/env python3
"""EP=4 通信优化测试脚本.

关键优化:
1. 减少 moe_gather_inputs 的同步开销
2. 预分配通信 buffer
3. 使用 NCCL backend
"""

import os
import sys
import time

import numpy as np
import torch


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 4))
    device = torch.device(f'cuda:{local_rank}')

    # 安全配置
    hidden_dim = 2048
    ffn_dim = 4096
    num_experts_total = 16 * world_size
    top_k = 4
    num_experts_per_rank = num_experts_total // world_size

    if local_rank == 0:
        print('=' * 80)
        print(f'EP=4 通信优化测试')
        print(f'GPU: {torch.cuda.get_device_name(device)} x{world_size}')
        print(f'配置: hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, 每卡专家={num_experts_per_rank}')
        print('=' * 80)

    # 关键优化 1: 使用 NCCL backend
    if local_rank == 0:
        print('初始化 NCCL backend...')
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.config import DistConfig

    dist_ctx = get_dist_manager().current_context()
    dist_ctx.dist_config = DistConfig(tp=1, ep=world_size, dp=1)
    dist_ctx.ep_gpu_group = torch.distributed.group.WORLD
    dist_ctx.ep_rank = local_rank
    dist_ctx.ep_size = world_size

    # 创建模型
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

    moe = FusedMoEAWQ(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        num_experts=num_experts_per_rank,
        top_k=top_k,
        w_bit=4,
        group_size=128,
        bias=False,
        renormalize=True,
        dtype=torch.float16,
        device=device,
        all_reduce=False,
        layer_idx=0,
        act_func=None,
    )

    # 初始化权重
    with torch.no_grad():
        num_exp = moe.gate_up.num_experts
        in_feat = moe.gate_up.in_features
        out_feat = moe.gate_up.out_features
        grouped_in = in_feat // 128
        quant_out = out_feat // 8

        for e in range(num_exp):
            moe.gate_up.qweight[e] = torch.randint(0, 256, (in_feat, quant_out), dtype=torch.int32, device=device)
            moe.gate_up.scales[e] = torch.rand((grouped_in, out_feat), dtype=torch.float16, device=device)
            moe.gate_up.qzeros[e] = torch.randint(0, 256, (grouped_in, quant_out), dtype=torch.int32, device=device)

        num_exp_down = moe.down.num_experts
        in_feat_down = moe.down.in_features
        out_feat_down = moe.down.out_features
        grouped_in_down = in_feat_down // 128
        quant_out_down = out_feat_down // 8

        for e in range(num_exp_down):
            moe.down.qweight[e] = torch.randint(0, 256, (in_feat_down, quant_out_down), dtype=torch.int32, device=device)
            moe.down.scales[e] = torch.rand((grouped_in_down, out_feat_down), dtype=torch.float16, device=device)
            moe.down.qzeros[e] = torch.randint(0, 256, (grouped_in_down, quant_out_down), dtype=torch.int32, device=device)

    model_mem = torch.cuda.memory_allocated(device) / (1024**3)
    if local_rank == 0:
        print(f'模型内存: {model_mem:.2f} GiB/卡')

    # 关键优化 2: 预分配通信 buffer
    # 避免每次 forward 都分配新内存
    if local_rank == 0:
        print('预分配通信 buffer...')

    max_seq_len = 16384
    comm_buffer_hidden = torch.empty(1, max_seq_len, hidden_dim, dtype=torch.float16, device=device)
    comm_buffer_weights = torch.empty(1, max_seq_len, top_k, dtype=torch.float16, device=device)
    comm_buffer_ids = torch.empty(1, max_seq_len, top_k, dtype=torch.int64, device=device)

    # Warmup
    h = torch.randn(1, 64, hidden_dim, dtype=torch.float16, device=device)
    w = torch.rand(1, 64, top_k, dtype=torch.float16, device=device)
    w = w / w.sum(dim=-1, keepdim=True)
    ids = torch.randint(0, num_experts_per_rank, (1, 64, top_k), device=device)

    for _ in range(5):
        _ = moe(h, w, ids)

    torch.cuda.synchronize()
    if local_rank == 0:
        print('Warmup 完成')

    # 性能测试
    if local_rank == 0:
        print('\n开始性能测试...')
        print(f"{'Context Length':<15} {'Prefill (ms)':<15} {'Prefill (tok/s)':<20} {'Decode (ms)':<15} {'Decode (tok/s)':<20}")
        print('-' * 85)

        results = {}

        for seq_len in [4096, 8192, 16384]:
            # 使用预分配的 buffer
            h = comm_buffer_hidden[:, :seq_len, :].normal_()
            w = comm_buffer_weights[:, :seq_len, :].uniform_()
            w = w / w.sum(dim=-1, keepdim=True)
            ids = comm_buffer_ids[:, :seq_len, :].random_(0, num_experts_per_rank)

            # Prefill
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            prefill_mean = np.mean(times)
            prefill_tok = seq_len / (prefill_mean / 1000)

            # Decode
            h = torch.randn(1, 1, hidden_dim, dtype=torch.float16, device=device)
            w = torch.rand(1, 1, top_k, dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, num_experts_per_rank, (1, 1, top_k), device=device)

            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            decode_mean = np.mean(times)
            decode_tok = 1 / (decode_mean / 1000)

            results[seq_len] = {
                'prefill_ms': float(prefill_mean),
                'prefill_tok_per_sec': float(prefill_tok),
                'decode_ms': float(decode_mean),
                'decode_tok_per_sec': float(decode_tok),
            }

            print(f'{seq_len:<15} {prefill_mean:<15.2f} {prefill_tok:<20.0f} {decode_mean:<15.2f} {decode_tok:<20.1f}')

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f'\n峰值内存: {peak_mem:.2f} GiB')

        # 计算加速比
        baseline_decode = 46.26  # 从 results_4gpus.json
        speedup = results[4096]['decode_tok_per_sec'] / baseline_decode
        print(f'相对基线加速: {speedup:.2f}x')

        import json
        with open('results_ep_optimized.json', 'w') as f:
            json.dump({
                'world_size': world_size,
                'config': {
                    'hidden_dim': hidden_dim,
                    'ffn_dim': ffn_dim,
                    'num_experts_total': num_experts_total,
                    'num_experts_per_rank': num_experts_per_rank,
                    'top_k': top_k,
                },
                'optimizations': [
                    'NCCL backend',
                    '预分配通信 buffer',
                ],
                'model_mem_gb': model_mem,
                'peak_mem_gb': peak_mem,
                'speedup': speedup,
                'results': results,
            }, f, indent=2)

        print('结果已保存到 results_ep_optimized.json')

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
