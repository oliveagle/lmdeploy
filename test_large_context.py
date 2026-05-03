#!/usr/bin/env python3
"""AWQ MoE EP=4 大上下文性能测试.

测试范围: 4k, 8k, 16k, 24k, 32k, ..., 512k
测量: prefill 速度, decode 速度
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


def main():
    parser = argparse.ArgumentParser(description='AWQ MoE EP 大上下文性能测试')
    parser.add_argument('--enable-cache', action='store_true', help='启用权重缓存')
    parser.add_argument('--cache-hot', type=int, default=16, help='缓存热点专家数')
    parser.add_argument('--min-seq', type=int, default=4096, help='最小序列长度')
    parser.add_argument('--max-seq', type=int, default=524288, help='最大序列长度 (512k)')
    parser.add_argument('--step', type=int, default=4096, help='序列长度步长')
    parser.add_argument('--prefill-iter', type=int, default=3, help='prefill 测试次数')
    parser.add_argument('--decode-iter', type=int, default=10, help='decode 测试次数')
    parser.add_argument('--output', type=str, default='large_context_results.json', help='输出文件')
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
        print('=' * 70)
        print('AWQ MoE EP=4 大上下文性能测试')
        print('=' * 70)
        print(f'GPU: {torch.cuda.get_device_name(device)} x{world_size}')
        print(f'配置: hidden={cfg["hidden_dim"]}, 专家={cfg["num_experts_per_rank"]}/卡')
        print(f'权重缓存: {args.enable_cache}, 热点专家: {args.cache_hot}')
        print(f'测试范围: {args.min_seq} - {args.max_seq} (步长: {args.step})')
        print('=' * 70)
        print()

    torch.cuda.empty_cache()

    # 创建模型
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

    # Warmup
    if local_rank == 0:
        print('Warmup...')
    for _ in range(2):
        h = torch.randn(1, 256, cfg['hidden_dim'], dtype=torch.float16, device=device)
        w = torch.rand(1, 256, cfg['top_k'], dtype=torch.float16, device=device)
        w = w / w.sum(dim=-1, keepdim=True)
        ids = torch.randint(0, cfg['num_experts_per_rank'], (1, 256, cfg['top_k']), device=device)
        _ = moe(h, w, ids)
        del h, w, ids

    torch.cuda.synchronize()

    # 生成测试序列长度列表: 4k, 8k, 16k, ..., 512k
    seq_lengths = []
    current = args.min_seq
    while current <= args.max_seq:
        seq_lengths.append(current)
        current *= 2  # 4k, 8k, 16k, 32k, ...

    if local_rank == 0:
        print(f'测试序列长度: {len(seq_lengths)} 个')
        print(f'范围: {seq_lengths[0]} - {seq_lengths[-1]}')
        print()
        print(f"{'Seq Len':<10} {'Prefill (ms)':<15} {'Prefill (tok/s)':<20} {'Decode (ms)':<15} {'Decode (tok/s)':<20} {'Status':<10}")
        print('-' * 100)

    results = {
        'config': cfg,
        'optimization': {
            'enable_cache': args.enable_cache,
            'cache_hot': args.cache_hot,
        },
        'model_mem_gb': model_mem,
        'seq_lengths': seq_lengths,
        'measurements': [],
    }

    for seq_len in seq_lengths:
        status = 'OK'

        try:
            # Prefill 测试
            h = torch.randn(1, seq_len, cfg['hidden_dim'], dtype=torch.float16, device=device)
            w = torch.rand(1, seq_len, cfg['top_k'], dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, cfg['num_experts_per_rank'], (1, seq_len, cfg['top_k']), device=device)

            prefill_times = []
            for _ in range(args.prefill_iter):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                prefill_times.append((time.perf_counter() - t0) * 1000)

            prefill_mean = np.mean(prefill_times)
            prefill_std = np.std(prefill_times)
            prefill_tok = seq_len / (prefill_mean / 1000)

            del h, w, ids
            torch.cuda.empty_cache()

            # Decode 测试
            h = torch.randn(1, 1, cfg['hidden_dim'], dtype=torch.float16, device=device)
            w = torch.rand(1, 1, cfg['top_k'], dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, cfg['num_experts_per_rank'], (1, 1, cfg['top_k']), device=device)

            decode_times = []
            for _ in range(args.decode_iter):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                decode_times.append((time.perf_counter() - t0) * 1000)

            decode_mean = np.mean(decode_times)
            decode_std = np.std(decode_times)
            decode_tok = 1 / (decode_mean / 1000)

            del h, w, ids
            torch.cuda.empty_cache()

            measurement = {
                'seq_len': seq_len,
                'prefill': {
                    'mean_ms': float(prefill_mean),
                    'std_ms': float(prefill_std),
                    'tok_per_sec': float(prefill_tok),
                },
                'decode': {
                    'mean_ms': float(decode_mean),
                    'std_ms': float(decode_std),
                    'tok_per_sec': float(decode_tok),
                },
                'status': 'OK',
            }
            results['measurements'].append(measurement)

            if local_rank == 0:
                print(f"{seq_len:<10} {prefill_mean:<15.2f} {prefill_tok:<20.0f} {decode_mean:<15.2f} {decode_tok:<20.1f} {status:<10}")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                status = 'OOM'
                measurement = {
                    'seq_len': seq_len,
                    'status': 'OOM',
                    'error': str(e),
                }
                results['measurements'].append(measurement)

                if local_rank == 0:
                    print(f"{seq_len:<10} {'-':<15} {'-':<20} {'-':<15} {'-':<20} {status:<10}")
                break  # OOM 后停止测试
            else:
                raise

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)

    if local_rank == 0:
        print()
        print(f'峰值内存: {peak_mem:.2f} GiB')
        print()

        # 计算加速比
        if args.min_seq in [m['seq_len'] for m in results['measurements'] if m['status'] == 'OK']:
            first_result = next(m for m in results['measurements'] if m['seq_len'] == args.min_seq and m['status'] == 'OK')
            if 'decode' in first_result:
                baseline = 46.26
                speedup = first_result['decode']['tok_per_sec'] / baseline
                print(f'相对基线加速: {speedup:.2f}x (seq_len={args.min_seq})')
                results['speedup'] = speedup

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
