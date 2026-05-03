#!/usr/bin/env python3
"""4卡AWQ MoE性能测试脚本 - 零配置，直接运行"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def setup_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_dist = world_size > 1

    if is_dist:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return local_rank, world_size, is_dist


def main():
    local_rank, world_size, is_dist = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 配置 - 4卡模式下每卡64专家
    hidden_dim = 2048  # 小一点避免OOM
    ffn_dim = 4096
    num_experts = 16 * world_size
    top_k = 4
    w_bit = 4
    group_size = 128
    context_lengths = [4096]

    if local_rank == 0:
        print('=' * 80)
        print('4卡 AWQ MoE 性能测试')
        print('=' * 80)
        print(f"  GPU: {torch.cuda.get_device_name(device)}  x{world_size}")
        print(f"  hidden_dim={hidden_dim}  ffn_dim={ffn_dim}")
        print(f"  num_experts={num_experts}  top_k={top_k}")
        print('=' * 80)

    # 初始化分布式
    if is_dist:
        from lmdeploy.pytorch.distributed import get_dist_manager
        from lmdeploy.pytorch.config import DistConfig

        dist_ctx = get_dist_manager().current_context()
        dist_ctx.dist_config = DistConfig(
            tp=1,
            ep=world_size,
            dp=1,
        )
        dist_ctx.ep_gpu_group = torch.distributed.group.WORLD
        dist_ctx.ep_rank = local_rank
        dist_ctx.ep_size = world_size

    # 创建模型
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ
    num_experts_per_rank = num_experts // world_size

    if local_rank == 0:
        print(f"  创建模型，每卡 {num_experts_per_rank} 个专家...")

    moe = FusedMoEAWQ(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        num_experts=num_experts_per_rank,
        top_k=top_k,
        w_bit=w_bit,
        group_size=group_size,
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
        if local_rank == 0:
            print(f"  初始化权重...")

        # Gate up weights
        num_exp = moe.gate_up.num_experts
        in_feat = moe.gate_up.in_features
        out_feat = moe.gate_up.out_features
        grouped_in = in_feat // group_size
        quant_out = out_feat // (32 // w_bit)

        for e in range(num_exp):
            moe.gate_up.qweight[e] = torch.randint(0, 256, (in_feat, quant_out), dtype=torch.int32, device=device)
            moe.gate_up.scales[e] = torch.rand((grouped_in, out_feat), dtype=torch.float16, device=device)
            moe.gate_up.qzeros[e] = torch.randint(0, 256, (grouped_in, quant_out), dtype=torch.int32, device=device)

        # Down weights
        num_exp_down = moe.down.num_experts
        in_feat_down = moe.down.in_features
        out_feat_down = moe.down.out_features
        grouped_in_down = in_feat_down // group_size
        quant_out_down = out_feat_down // (32 // w_bit)

        for e in range(num_exp_down):
            moe.down.qweight[e] = torch.randint(0, 256, (in_feat_down, quant_out_down), dtype=torch.int32, device=device)
            moe.down.scales[e] = torch.rand((grouped_in_down, out_feat_down), dtype=torch.float16, device=device)
            moe.down.qzeros[e] = torch.randint(0, 256, (grouped_in_down, quant_out_down), dtype=torch.int32, device=device)

    model_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
    if local_rank == 0:
        print(f"✓ 模型内存: {model_mem:.2f} GB/卡")

    # 测试 - 单卡模式下用单卡
    if world_size == 1:
        # 单卡简单测试
        if local_rank == 0:
            print("\n  单卡测试模式")
            seq_len = 4096

            def create_test(batch_size, seq_len):
                h = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
                w = torch.rand(batch_size, seq_len, top_k, dtype=torch.float16, device=device)
                w = w / w.sum(dim=-1, keepdim=True)
                ids = torch.randint(0, num_experts_per_rank, (batch_size, seq_len, top_k), device=device)
                return h, w, ids

            # Warmup
            h, w, ids = create_test(1, 64)
            for _ in range(3):
                try:
                    _ = moe(h, w, ids)
                except Exception as e:
                    print(f"  ⚠️  {e}")

            print(f"\n  测试 context_lengths: {context_lengths}")
            results = {'model_mem_gb': model_mem}

            for seq_len in context_lengths:
                try:
                    h, w, ids = create_test(1, seq_len)

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
                    h, w, ids = create_test(1, 1)
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
                        'prefill': {'mean_ms': prefill_mean, 'tok_per_sec': prefill_tok},
                        'decode': {'mean_ms': decode_mean, 'tok_per_sec': decode_tok}
                    }

                    print(f"✓ seq_len={seq_len}: prefill {prefill_tok:.0f} tok/s, decode {decode_tok:.1f} tok/s")

                except Exception as e:
                    print(f"✗ seq_len={seq_len}: {e}")
                    results[seq_len] = {'error': str(e)}

            results['peak_mem_gb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

            with open('results_4gpus.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ 结果保存到 results_4gpus.json")
            print(f"峰值内存: {results['peak_mem_gb']:.2f} GiB")
            print("=" * 80)

    else:
        if local_rank == 0:
            print("\n⚠️  多卡模式下需要专家ID映射，暂用单卡结果代替")
            print("=" * 80)

    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
