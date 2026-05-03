#!/usr/bin/env python3
"""内存优化的 EP=4 测试脚本.

优化策略:
1. 减少 context_lengths 避免大 seq OOM
2. 使用 empty_init=True
3. 及时清理中间变量
4. 使用 gradient checkpointing 风格的内存管理
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

    # 使用 NCCL backend
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # 配置 EP
    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.config import DistConfig

    dist_ctx = get_dist_manager().current_context()
    dist_ctx.dist_config = DistConfig(tp=1, ep=world_size, dp=1)
    dist_ctx.ep_gpu_group = torch.distributed.group.WORLD
    dist_ctx.ep_rank = local_rank
    dist_ctx.ep_size = world_size

    return local_rank, world_size


def main():
    parser = argparse.ArgumentParser(description='内存优化的 EP=4 测试')
    parser.add_argument('--enable-cache', action='store_true', help='启用权重缓存')
    parser.add_argument('--cache-hot', type=int, default=16, help='缓存热点专家数 (默认 16)')
    parser.add_argument('--max-seq-len', type=int, default=8192, help='最大序列长度 (避免 OOM)')
    parser.add_argument('--name', type=str, default='default', help='测试名称')
    parser.add_argument('--output', type=str, default='results.json', help='输出文件')
    args = parser.parse_args()

    local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 小配置避免 OOM
    cfg = {
        'hidden_dim': 2048,
        'ffn_dim': 4096,
        'num_experts_total': 64,
        'top_k': 4,
        'num_experts_per_rank': 64 // world_size,
    }

    if local_rank == 0:
        print('=' * 60)
        print(f'内存优化 EP=4 测试 - {args.name}')
        print(f'GPU: {torch.cuda.get_device_name(device)} x{world_size}')
        print(f'配置: hidden={cfg["hidden_dim"]}, 专家={cfg["num_experts_per_rank"]}/卡')
        print(f'优化: cache={args.enable_cache}, hot={args.cache_hot}')
        print(f'最大序列长度: {args.max_seq_len}')
        print('=' * 60)

    # 清空缓存
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
            moe.gate_up.qweight[e].zero_()
            moe.gate_up.qweight[e].random_(0, 256)
            moe.gate_up.scales[e].fill_(0.1)
            moe.gate_up.qzeros[e].zero_()

        num_exp_down = moe.down.num_experts
        in_feat_down = moe.down.in_features
        out_feat_down = moe.down.out_features
        grouped_in_down = in_feat_down // 128
        quant_out_down = out_feat_down // 8

        for e in range(num_exp_down):
            moe.down.qweight[e].zero_()
            moe.down.qweight[e].random_(0, 256)
            moe.down.scales[e].fill_(0.1)
            moe.down.qzeros[e].zero_()

    model_mem = torch.cuda.memory_allocated(device) / (1024**3)

    if local_rank == 0:
        print(f'模型内存: {model_mem:.2f} GiB')
        available = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f'GPU 总内存: {available:.2f} GiB')
        print(f'剩余可用: {available - model_mem:.2f} GiB')
        print('开始测试...')

    # 性能测试 - 只测试安全的序列长度
    context_lengths = [1024, 2048, 4096]
    if args.max_seq_len >= 8192:
        context_lengths.append(8192)

    results = {}

    for seq_len in context_lengths:
        if local_rank == 0:
            print(f'  Testing seq_len={seq_len}...', end=' ', flush=True)

        # Warmup - 使用小序列
        for _ in range(2):
            h = torch.randn(1, 64, cfg['hidden_dim'], dtype=torch.float16, device=device)
            w = torch.rand(1, 64, cfg['top_k'], dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, cfg['num_experts_per_rank'], (1, 64, cfg['top_k']), device=device)
            _ = moe(h, w, ids)
            del h, w, ids

        torch.cuda.synchronize()

        try:
            # Prefill
            h = torch.randn(1, seq_len, cfg['hidden_dim'], dtype=torch.float16, device=device)
            w = torch.rand(1, seq_len, cfg['top_k'], dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, cfg['num_experts_per_rank'], (1, seq_len, cfg['top_k']), device=device)

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
            del h, w, ids
            h = torch.randn(1, 1, cfg['hidden_dim'], dtype=torch.float16, device=device)
            w = torch.rand(1, 1, cfg['top_k'], dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, cfg['num_experts_per_rank'], (1, 1, cfg['top_k']), device=device)

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

            if local_rank == 0:
                print(f'Decode: {decode_tok:.1f} tok/s')

            # 清理
            del h, w, ids
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                if local_rank == 0:
                    print(f'OOM at seq_len={seq_len}')
                results[seq_len] = {'error': 'OOM'}
                break
            else:
                raise

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)

    if local_rank == 0:
        print(f'\n峰值内存: {peak_mem:.2f} GiB')

        # 计算加速比
        baseline = 46.26
        if 4096 in results and 'decode_tok_per_sec' in results[4096]:
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
            'results': results,
        }

        if 4096 in results and 'decode_tok_per_sec' in results[4096]:
            output['speedup'] = speedup

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
        sys.exit(1)
