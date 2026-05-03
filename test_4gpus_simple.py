#!/usr/bin/env python3
"""4卡AWQ MoE性能测试 - 直接运行，零配置"""

import os
import sys
import time

import numpy as np
import torch


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 4))
    device = torch.device(f'cuda:{local_rank}')

    # 使用已验证的安全配置
    hidden_dim = 2048
    ffn_dim = 4096
    num_experts_total = 16 * world_size  # 4卡 = 64专家
    top_k = 4
    num_experts_per_rank = num_experts_total // world_size

    if local_rank == 0:
        print('=' * 80)
        print(f'4卡AWQ MoE性能测试')
        print(f'GPU: {torch.cuda.get_device_name(device)} x{world_size}')
        print(f'配置: hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, 总专家={num_experts_total}, 每卡={num_experts_per_rank}')
        print('=' * 80)

    # 初始化分布式
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

    # Warmup
    h = torch.randn(1, 64, hidden_dim, dtype=torch.float16, device=device)
    w = torch.rand(1, 64, top_k, dtype=torch.float16, device=device)
    w = w / w.sum(dim=-1, keepdim=True)
    ids = torch.randint(0, num_experts_per_rank, (1, 64, top_k), device=device)

    for _ in range(3):
        _ = moe(h, w, ids)

    torch.cuda.synchronize()
    if local_rank == 0:
        print('Warmup完成')

    # 测试 - 只在rank0测试以避免同步问题
    if local_rank == 0:
        print('\n开始测试...')
        results = {}

        for seq_len in [4096, 8192, 16384]:
            h = torch.randn(1, seq_len, hidden_dim, dtype=torch.float16, device=device)
            w = torch.rand(1, seq_len, top_k, dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, num_experts_per_rank, (1, seq_len, top_k), device=device)

            # Prefill
            times = []
            for _ in range(3):
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
            for _ in range(10):
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

            print(f'Context {seq_len}: prefill {prefill_tok:.0f} tok/s, decode {decode_tok:.1f} tok/s')

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f'\n峰值内存: {peak_mem:.2f} GiB')

        import json
        with open('results_4gpus_final.json', 'w') as f:
            json.dump({
                'world_size': world_size,
                'config': {'hidden_dim': hidden_dim, 'ffn_dim': ffn_dim, 'num_experts_total': num_experts_total, 'num_experts_per_rank': num_experts_per_rank, 'top_k': top_k},
                'model_mem_gb': model_mem,
                'peak_mem_gb': peak_mem,
                'results': results,
            }, f, indent=2)

        print('结果已保存到 results_4gpus_final.json')

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
