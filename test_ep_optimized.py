#!/usr/bin/env python3
"""AWQ MoE EP=4 性能优化综合测试.

优化策略:
1. NCCL backend
2. 权重缓存 (partial)
3. 预分配 buffer
4. 减少同步点

测试方法:
1. 基线测试 (无优化)
2. 逐项优化测试
3. 组合优化测试
"""

import argparse
import json
import os
import sys
import time

import torch


def setup_optimized_environment():
    """配置优化的运行环境."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 4))

    # 设置 NCCL backend
    os.environ['NCCL_ALGO'] = 'Ring'  # V100 上 NCCL 最优
    os.environ['NCCL_PROTO'] = 'Simple'

    # 初始化分布式
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # 配置 EP
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

    return local_rank, world_size


def create_optimized_moe(cfg, device, enable_cache=False, cache_hot=0):
    """创建优化后的 MoE 模型."""
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

    return FusedMoEAWQ(
        hidden_dim=cfg['hidden_dim'],
        ffn_dim=cfg['ffn_dim'],
        num_experts=cfg['num_experts_per_rank'],
        top_k=cfg['top_k'],
        w_bit=4,
        group_size=128,
        bias=False,
        renormalize=True,
        dtype=torch.float16,
        device=device,
        all_reduce=False,
        layer_idx=0,
        act_func=None,
        enable_weight_cache=enable_cache,
        cache_hot_experts=cache_hot if enable_cache else None,
    )


def init_moe_weights(moe, device, group_size=128):
    """初始化 MoE 权重."""
    with torch.no_grad():
        # Gate/Up
        num_exp = moe.gate_up.num_experts
        in_feat = moe.gate_up.in_features
        out_feat = moe.gate_up.out_features
        grouped_in = in_feat // group_size
        quant_out = out_feat // 8

        for e in range(num_exp):
            moe.gate_up.qweight[e].zero_()
            moe.gate_up.qweight[e].random_(0, 256)
            moe.gate_up.scales[e].fill_(0.1)
            moe.gate_up.qzeros[e].zero_()

        # Down
        num_exp_down = moe.down.num_experts
        in_feat_down = moe.down.in_features
        out_feat_down = moe.down.out_features
        grouped_in_down = in_feat_down // group_size
        quant_out_down = out_feat_down // 8

        for e in range(num_exp_down):
            moe.down.qweight[e].zero_()
            moe.down.qweight[e].random_(0, 256)
            moe.down.scales[e].fill_(0.1)
            moe.down.qzeros[e].zero_()


def benchmark_moe(moe, cfg, device, seq_len, num_prefill_iter=5, num_decode_iter=20):
    """性能测试."""
    hidden_dim = cfg['hidden_dim']
    top_k = cfg['top_k']
    num_exp_per_rank = cfg['num_experts_per_rank']

    # Warmup
    for _ in range(3):
        h = torch.randn(1, 64, hidden_dim, dtype=torch.float16, device=device)
        w = torch.rand(1, 64, top_k, dtype=torch.float16, device=device)
        w = w / w.sum(dim=-1, keepdim=True)
        ids = torch.randint(0, num_exp_per_rank, (1, 64, top_k), device=device)
        _ = moe(h, w, ids)

    torch.cuda.synchronize()

    # Prefill
    h = torch.randn(1, seq_len, hidden_dim, dtype=torch.float16, device=device)
    w = torch.rand(1, seq_len, top_k, dtype=torch.float16, device=device)
    w = w / w.sum(dim=-1, keepdim=True)
    ids = torch.randint(0, num_exp_per_rank, (1, seq_len, top_k), device=device)

    times = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_prefill_iter):
        _ = moe(h, w, ids)
    torch.cuda.synchronize()
    prefill_mean = (time.perf_counter() - start) * 1000 / num_prefill_iter

    # Decode
    h = torch.randn(1, 1, hidden_dim, dtype=torch.float16, device=device)
    w = torch.rand(1, 1, top_k, dtype=torch.float16, device=device)
    w = w / w.sum(dim=-1, keepdim=True)
    ids = torch.randint(0, num_exp_per_rank, (1, 1, top_k), device=device)

    times = []
    for _ in range(num_decode_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = moe(h, w, ids)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    decode_mean = sum(times) / len(times)
    decode_tok = 1 / (decode_mean / 1000)

    return prefill_mean, decode_tok


def main():
    parser = argparse.ArgumentParser(description='AWQ MoE EP=4 优化测试')
    parser.add_argument('--enable-cache', action='store_true', help='启用权重缓存')
    parser.add_argument('--cache-hot', type=int, default=32, help='缓存热点专家数')
    parser.add_argument('--name', type=str, default='default', help='测试名称')
    parser.add_argument('--output', type=str, default='results.json', help='输出文件')
    args = parser.parse_args()

    local_rank, world_size = setup_optimized_environment()
    device = torch.device(f'cuda:{local_rank}')

    # 配置 (使用小配置快速测试)
    cfg = {
        'hidden_dim': 2048,
        'ffn_dim': 4096,
        'num_experts_total': 64,
        'top_k': 4,
        'num_experts_per_rank': 64 // world_size,
    }

    if local_rank == 0:
        print('=' * 60)
        print(f'EP=4 性能优化测试 - {args.name}')
        print(f'GPU: {torch.cuda.get_device_name(device)} x{world_size}')
        print(f'配置: hidden={cfg["hidden_dim"]}, ffn={cfg["ffn_dim"]}, 专家={cfg["num_experts_per_rank"]}/卡')
        print(f'优化: cache={args.enable_cache}, hot={args.cache_hot}')
        print('=' * 60)

    # 创建模型
    moe = create_optimized_moe(cfg, device, args.enable_cache, args.cache_hot)
    init_moe_weights(moe, device)

    model_mem = torch.cuda.memory_allocated(device) / (1024**3)

    if local_rank == 0:
        print(f'模型内存: {model_mem:.2f} GiB')
        print('开始测试...')

    # 性能测试
    results = {}
    for seq_len in [4096, 8192]:
        if local_rank == 0:
            print(f'  Testing seq_len={seq_len}...', end=' ')

        prefill_ms, decode_tok = benchmark_moe(moe, cfg, device, seq_len)
        results[seq_len] = {
            'prefill_ms': prefill_ms,
            'decode_tok_per_sec': decode_tok,
        }

        if local_rank == 0:
            print(f'Decode: {decode_tok:.1f} tok/s')

    peak_mem = torch.cuda.memory_allocated(device) / (1024**3)

    if local_rank == 0:
        print(f'峰值内存: {peak_mem:.2f} GiB')

        # 计算加速比
        baseline = 46.26  # 从 results_4gpus.json
        speedup = results[4096]['decode_tok_per_sec'] / baseline
        print(f'相对基线加速: {speedup:.2f}x')

        # 保存结果
        output = {
            'test_name': args.name,
            'optimization': {
                'enable_cache': args.enable_cache,
                'cache_hot': args.cache_hot,
            },
            'config': cfg,
            'model_mem_gb': model_mem,
            'peak_mem_gb': peak_mem,
            'baseline_decode_tok_per_sec': baseline,
            'speedup': speedup,
            'results': results,
        }

        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

        print(f'结果已保存: {args.output}')

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
