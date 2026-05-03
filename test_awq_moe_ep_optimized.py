#!/usr/bin/env python3
"""AWQ MoE EP 性能优化测试脚本.

优化策略:
1. 权重预解量化缓存 (enable_weight_cache)
2. 分层缓存 (cache_hot_experts)
3. 批处理优化
4. 混合精度推理
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


# ============================================================
# Qwen3.6-35B-A3B-AWQ 真实配置
# ============================================================
REAL_CONFIG = dict(
    hidden_dim=5120,
    ffn_dim=13824,
    num_experts=256,
    top_k=8,
    w_bit=4,
    group_size=128,
)


def parse_args():
    """解析参数."""
    p = argparse.ArgumentParser(description='AWQ MoE EP Performance Optimization Test')
    p.add_argument('--context-lengths', nargs='+', type=int, default=[4096, 8192, 16384])
    p.add_argument('--num-iterations-prefill', type=int, default=5)
    p.add_argument('--num-iterations-decode', type=int, default=50)
    p.add_argument('--output', type=str, default='awq_moe_ep_optimized_results.json')
    p.add_argument('--batch-size', type=int, default=1)

    # 优化参数
    p.add_argument('--enable-weight-cache', action='store_true',
                    help='Enable weight pre-dequantization cache')
    p.add_argument('--cache-hot-experts', type=int, default=None,
                    help='Number of hot experts to cache (None = all)')
    p.add_argument('--use-fp32-dequant', action='store_true',
                    help='Use FP32 for dequantization (more accurate)')
    p.add_argument('--enable-nccl', action='store_true',
                    help='Use NCCL for EP communication')

    # 模型参数
    for k in ('hidden_dim', 'ffn_dim', 'num_experts', 'top_k', 'w_bit', 'group_size'):
        env = os.environ.get(k.upper())
        p.add_argument(f'--{k}', type=int, default=int(env) if env else None)

    args = p.parse_args()
    return args


def setup_distributed():
    """初始化分布式环境."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_dist = world_size > 1

    if is_dist:
        torch.distributed.init_process_group(backend='nccl' if args.enable_nccl else 'gloo')
        torch.cuda.set_device(local_rank)

    return local_rank, world_size, is_dist


def resolve_config(args, world_size):
    """确定最终模型配置."""
    cfg = {}
    for k in ('hidden_dim', 'ffn_dim', 'num_experts', 'top_k', 'w_bit', 'group_size'):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
        elif world_size > 1:
            cfg[k] = REAL_CONFIG[k]
        else:
            cfg[k] = {
                'hidden_dim': 2048,
                'ffn_dim': 4096,
                'num_experts': 16,
                'top_k': 2,
                'w_bit': 4,
                'group_size': 128,
            }[k]

    if world_size > 1:
        assert cfg['num_experts'] % world_size == 0
        cfg['per_rank_experts'] = cfg['num_experts'] // world_size

    return cfg


def create_test_inputs(batch_size, seq_len, hidden_dim, num_experts, top_k, device):
    """创建随机输入."""
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    weights = torch.rand(batch_size, seq_len, top_k, dtype=torch.float16, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = torch.randint(0, num_experts, (batch_size, seq_len, top_k), device=device)
    return hidden, weights, ids


def benchmark_method(moe, hidden_states, topk_weights, topk_ids, num_iterations, method_name, local_rank=0):
    """性能测试某个方法."""
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = moe(hidden_states, topk_weights, topk_ids)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = moe(hidden_states, topk_weights, topk_ids)
        torch.cuda.synchronize()

    end = time.perf_counter()
    avg_time = (end - start) * 1000 / num_iterations

    if local_rank == 0:
        print(f"  {method_name}: {avg_time:.2f} ms")

    return avg_time


def main():
    """主函数."""
    args = parse_args()
    local_rank, world_size, is_dist = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    cfg = resolve_config(args, world_size)

    if local_rank == 0:
        print('=' * 80)
        print('AWQ MoE EP Performance Optimization Test')
        print('=' * 80)
        print(f"  GPU: {torch.cuda.get_device_name(device)} x{world_size}")
        print(f"  hidden_dim={cfg['hidden_dim']}  ffn_dim={cfg['ffn_dim']}")
        print(f"  num_experts={cfg['num_experts']}  top_k={cfg['top_k']}")
        print(f"  Optimization:")
        print(f"    enable_weight_cache={args.enable_weight_cache}")
        print(f"    cache_hot_experts={args.cache_hot_experts}")
        print(f"    use_fp32_dequant={args.use_fp32_dequant}")
        print(f"    enable_nccl={args.enable_nccl}")
        print('=' * 80)

    # 初始化分布式配置
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

    num_experts_per_rank = cfg['num_experts'] // world_size if is_dist else cfg['num_experts']

    # 创建模型
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

    if local_rank == 0:
        print(f"  Creating model with {num_experts_per_rank} experts...")

    moe = FusedMoEAWQ(
        hidden_dim=cfg['hidden_dim'],
        ffn_dim=cfg['ffn_dim'],
        num_experts=num_experts_per_rank,
        top_k=cfg['top_k'],
        w_bit=cfg['w_bit'],
        group_size=cfg['group_size'],
        bias=False,
        renormalize=True,
        dtype=torch.float16,
        device=device,
        all_reduce=False,
        layer_idx=0,
        act_func=None,
        enable_weight_cache=args.enable_weight_cache,
        cache_hot_experts=args.cache_hot_experts,
    )

    # 初始化权重
    with torch.no_grad():
        num_exp = moe.gate_up.num_experts
        in_feat = moe.gate_up.in_features
        out_feat = moe.gate_up.out_features
        grouped_in = in_feat // cfg['group_size']
        quant_out = out_feat // (32 // cfg['w_bit'])

        for e in range(num_exp):
            moe.gate_up.qweight[e] = torch.randint(
                0, 256, (in_feat, quant_out), dtype=torch.int32, device=device
            )
            moe.gate_up.scales[e] = torch.rand(
                (grouped_in, out_feat), dtype=torch.float16, device=device
            )
            moe.gate_up.qzeros[e] = torch.randint(
                0, 256, (grouped_in, quant_out), dtype=torch.int32, device=device
            )

        num_exp_down = moe.down.num_experts
        in_feat_down = moe.down.in_features
        out_feat_down = moe.down.out_features
        grouped_in_down = in_feat_down // cfg['group_size']
        quant_out_down = out_feat_down // (32 // cfg['w_bit'])

        for e in range(num_exp_down):
            moe.down.qweight[e] = torch.randint(
                0, 256, (in_feat_down, quant_out_down), dtype=torch.int32, device=device
            )
            moe.down.scales[e] = torch.rand(
                (grouped_in_down, out_feat_down), dtype=torch.float16, device=device
            )
            moe.down.qzeros[e] = torch.randint(
                0, 256, (grouped_in_down, quant_out_down), dtype=torch.int32, device=device
            )

    model_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
    if local_rank == 0:
        print(f"  Model memory: {model_mem:.2f} GB/GPU")

    # 性能测试
    results = {
        'config': cfg,
        'optimization': {
            'enable_weight_cache': args.enable_weight_cache,
            'cache_hot_experts': args.cache_hot_experts,
            'use_fp32_dequant': args.use_fp32_dequant,
            'enable_nccl': args.enable_nccl,
        },
        'model_mem_gb': model_mem,
    }

    for seq_len in args.context_lengths:
        if local_rank == 0:
            print(f"\n{'=' * 80}")
            print(f"Testing seq_len={seq_len}")
            print(f"{'=' * 80}")

        # Warmup
        h, w, ids = create_test_inputs(1, seq_len, cfg['hidden_dim'], num_experts_per_rank, cfg['top_k'], device)
        for _ in range(3):
            _ = moe(h, w, ids)
        torch.cuda.synchronize()

        try:
            # Prefill
            if local_rank == 0:
                print("  Prefill...")
            prefill_times = []
            for i in range(args.num_iterations_prefill):
                h, w, ids = create_test_inputs(1, seq_len, cfg['hidden_dim'], num_experts_per_rank, cfg['top_k'], device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                prefill_times.append((time.perf_counter() - t0) * 1000)

            prefill_mean = np.mean(prefill_times)
            prefill_tok_per_sec = seq_len / (prefill_mean / 1000)

            # Decode
            if local_rank == 0:
                print("  Decode...")
            decode_times = []
            for i in range(args.num_iterations_decode):
                h, w, ids = create_test_inputs(1, 1, cfg['hidden_dim'], num_experts_per_rank, cfg['top_k'], device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                decode_times.append((time.perf_counter() - t0) * 1000)

            decode_mean = np.mean(decode_times)
            decode_tok_per_sec = 1 / (decode_mean / 1000)

            results[str(seq_len)] = {
                'prefill': {'mean_ms': prefill_mean, 'tok_per_sec': prefill_tok_per_sec},
                'decode': {'mean_ms': decode_mean, 'tok_per_sec': decode_tok_per_sec}
            }

            if local_rank == 0:
                print(f"  Prefill: {prefill_tok_per_sec:.0f} tok/s ({prefill_mean:.2f} ms)")
                print(f"  Decode:  {decode_tok_per_sec:.1f} tok/s ({decode_mean:.2f} ms)")

        except Exception as e:
            if local_rank == 0:
                print(f"  Error: {e}")
            results[str(seq_len)] = {'error': str(e)}

    if local_rank == 0:
        results['peak_mem_gb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        print(f"Peak memory: {results['peak_mem_gb']:.2f} GiB")

    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
