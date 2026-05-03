#!/usr/bin/env python3
"""纯净 MoE 层性能测试 - 无 KV Cache，无前缀缓存干扰。

直接测试 MoE 层的计算性能:
1. 禁用 KV Cache
2. 每次使用不同的随机输入 (避免前缀缓存命中)
3. 强制 CUDA 同步确保准确计时
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def setup_distributed():
    """初始化分布式环境."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 4))

    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.config import DistConfig

    dist_ctx = get_dist_manager().current_context()
    dist_ctx.dist_config = DistConfig(tp=1, ep=world_size, dp=1)
    dist_ctx.ep_gpu_group = torch.distributed.group.WORLD
    dist_ctx.ep_rank = local_rank
    dist_ctx.ep_size = world_size

    return local_rank, world_size


def test_moe_layer(moe, cfg, device, seq_len, num_iter, local_rank):
    """测试 MoE 层性能 - 纯计算，无缓存干扰。

    关键优化:
    1. 每次迭代使用不同的随机输入 (避免任何缓存)
    2. 强制 CUDA 同步
    3. 不使用 KV Cache
    """
    hidden_dim = cfg['hidden_dim']
    top_k = cfg['top_k']
    num_exp_per_rank = cfg['num_experts_per_rank']

    times = []

    for i in range(num_iter):
        # 每次使用不同的随机输入 (防止前缀缓存命中)
        h = torch.randn(1, seq_len, hidden_dim, dtype=torch.float16, device=device)
        w = torch.rand(1, seq_len, top_k, dtype=torch.float16, device=device)
        w = w / w.sum(dim=-1, keepdim=True)

        # 确保专家 ID 随机分布 (防止路由缓存)
        ids = torch.randint(0, num_exp_per_rank, (1, seq_len, top_k), device=device)

        # 强制同步，确保准确计时
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 纯 MoE 计算，无 KV Cache
        _ = moe(h, w, ids)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)  # 转换为毫秒

        # 显式释放，防止内存累积
        del h, w, ids

    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser(description='纯净 MoE 层性能测试')
    parser.add_argument('--enable-cache', action='store_true', help='启用权重缓存')
    parser.add_argument('--cache-hot', type=int, default=16, help='缓存热点专家数')
    parser.add_argument('--max-seq', type=int, default=524288, help='最大序列长度')
    parser.add_argument('--iter', type=int, default=5, help='每个序列长度的测试次数')
    parser.add_argument('--output', type=str, default='pure_moe_results.json', help='输出文件')
    args = parser.parse_args()

    local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 配置
    cfg = {
        'hidden_dim': 2048,
        'ffn_dim': 4096,
        'num_experts_total': 64,
        'top_k': 4,
        'num_experts_per_rank': 64 // world_size,
    }

    if local_rank == 0:
        print('=' * 80)
        print('纯净 MoE 层性能测试 (无 KV Cache，无前缀缓存)')
        print('=' * 80)
        print(f'GPU: {torch.cuda.get_device_name(device)} x{world_size}')
        print(f'配置: hidden={cfg["hidden_dim"]}, 专家={cfg["num_experts_per_rank"]}/卡')
        print(f'权重缓存: {args.enable_cache}, 热点专家: {args.cache_hot}')
        print(f'测试次数: {args.iter} 次/序列长度')
        print('=' * 80)
        print()

    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 创建 MoE 层 (不创建完整模型，避免 KV Cache)
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

    moe = FusedMoEAWQ(
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
        enable_weight_cache=args.enable_cache,
        cache_hot_experts=args.cache_hot if args.enable_cache else None,
    )

    # 初始化权重
    with torch.no_grad():
        num_exp = moe.gate_up.num_experts
        in_feat = moe.gate_up.in_features
        out_feat = moe.gate_up.out_features
        grouped_in = in_feat // 128
        quant_out = out_feat // 8

        for e in range(num_exp):
            moe.gate_up.qweight[e].random_(0, 256)
            moe.gate_up.scales[e].fill_(0.1)
            moe.gate_up.qzeros[e].zero_()

        num_exp_down = moe.down.num_experts
        in_feat_down = moe.down.in_features
        out_feat_down = moe.down.out_features
        grouped_in_down = in_feat_down // 128
        quant_out_down = out_feat_down // 8

        for e in range(num_exp_down):
            moe.down.qweight[e].random_(0, 256)
            moe.down.scales[e].fill_(0.1)
            moe.down.qzeros[e].zero_()

    model_mem = torch.cuda.memory_allocated(device) / (1024**3)

    if local_rank == 0:
        print(f'模型内存: {model_mem:.2f} GiB')
        print()
        print(f"{'Seq Len':<12} {'Latency (ms)':<15} {'Std (ms)':<12} {'Tokens/s':<15} {'Status':<10}")
        print('-' * 80)

    # Warmup (使用不同的输入)
    if local_rank == 0:
        print('Warmup...')

    for _ in range(3):
        h = torch.randn(1, 256, cfg['hidden_dim'], dtype=torch.float16, device=device)
        w = torch.rand(1, 256, cfg['top_k'], dtype=torch.float16, device=device)
        w = w / w.sum(dim=-1, keepdim=True)
        ids = torch.randint(0, cfg['num_experts_per_rank'], (1, 256, cfg['top_k']), device=device)
        _ = moe(h, w, ids)
        del h, w, ids

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 测试序列长度: 4k, 8k, 16k, ..., max_seq
    seq_lengths = []
    current = 4096
    while current <= args.max_seq:
        seq_lengths.append(current)
        current *= 2

    results = {
        'config': cfg,
        'optimization': {
            'enable_cache': args.enable_cache,
            'cache_hot': args.cache_hot,
        },
        'model_mem_gb': model_mem,
        'test_iterations': args.iter,
        'note': 'Pure MoE layer test - no KV cache, no prefix cache interference',
        'measurements': [],
    }

    for seq_len in seq_lengths:
        try:
            # 测试 MoE 层性能
            mean_ms, std_ms = test_moe_layer(moe, cfg, device, seq_len, args.iter, local_rank)

            tokens_per_sec = seq_len / (mean_ms / 1000)

            measurement = {
                'seq_len': seq_len,
                'latency_ms': float(mean_ms),
                'std_ms': float(std_ms),
                'tokens_per_sec': float(tokens_per_sec),
                'status': 'OK',
            }
            results['measurements'].append(measurement)

            if local_rank == 0:
                print(f"{seq_len:<12} {mean_ms:<15.2f} {std_ms:<12.2f} {tokens_per_sec:<15.0f} OK")

            # 清理缓存
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                if local_rank == 0:
                    print(f"{seq_len:<12} {'-':<15} {'-':<12} {'-':<15} OOM")
                results['measurements'].append({
                    'seq_len': seq_len,
                    'status': 'OOM',
                    'error': str(e),
                })
                break
            else:
                raise

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)

    if local_rank == 0:
        print()
        print(f'峰值内存: {peak_mem:.2f} GiB')
        print()

        # 计算加速比 (相对于基线 4k)
        if results['measurements'] and results['measurements'][0]['status'] == 'OK':
            baseline_tok = results['measurements'][0]['tokens_per_sec']
            baseline_46 = 46.26  # 之前的基线

            print(f'当前配置 4k 吞吐: {baseline_tok:.1f} tokens/s')
            print(f'相对之前基线加速: {baseline_tok / baseline_46:.2f}x')
            print()

            # 计算扩展性 (4k vs 64k)
            for m in results['measurements']:
                if m['seq_len'] == 65536 and m['status'] == 'OK':
                    scalability = baseline_tok / m['tokens_per_sec']
                    print(f'扩展性 (4k vs 64k): {scalability:.2f}x (理想值 1.0)')
                    break

        # 保存结果
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'结果已保存: {args.output}')

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
