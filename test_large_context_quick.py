#!/usr/bin/env python3
"""快速大上下文测试 - 单卡版本."""

import argparse
import json
import time
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-cache', action='store_true')
    parser.add_argument('--cache-hot', type=int, default=16)
    parser.add_argument('--max-seq', type=int, default=131072)  # 128k
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 小配置
    cfg = {
        'hidden_dim': 2048,
        'ffn_dim': 4096,
        'num_experts': 16,
        'top_k': 4,
    }

    print('=' * 60)
    print(f'快速大上下文测试 (单卡)')
    print(f'GPU: {torch.cuda.get_device_name(device)}')
    print(f'权重缓存: {args.enable_cache}, 热点: {args.cache_hot}')
    print('=' * 60)

    # 创建模型 (无 EP)
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

    moe = FusedMoEAWQ(
        hidden_dim=cfg['hidden_dim'],
        ffn_dim=cfg['ffn_dim'],
        num_experts=cfg['num_experts'],
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

    print(f'模型内存: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GiB')
    print()

    # Warmup
    print('Warmup...')
    for _ in range(2):
        h = torch.randn(1, 256, cfg['hidden_dim'], dtype=torch.float16, device=device)
        w = torch.rand(1, 256, cfg['top_k'], dtype=torch.float16, device=device)
        w = w / w.sum(dim=-1, keepdim=True)
        ids = torch.randint(0, cfg['num_experts'], (1, 256, cfg['top_k']), device=device)
        _ = moe(h, w, ids)
        del h, w, ids

    torch.cuda.synchronize()

    # 测试序列长度: 4k, 8k, 16k, ..., max_seq
    seq_lengths = []
    current = 4096
    while current <= args.max_seq:
        seq_lengths.append(current)
        current *= 2

    print(f'测试序列长度: {seq_lengths}')
    print()
    print(f"{'Seq Len':<10} {'Prefill (tok/s)':<20} {'Decode (tok/s)':<20} {'Status':<10}")
    print('-' * 60)

    results = []

    for seq_len in seq_lengths:
        try:
            # Prefill
            h = torch.randn(1, seq_len, cfg['hidden_dim'], dtype=torch.float16, device=device)
            w = torch.rand(1, seq_len, cfg['top_k'], dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, cfg['num_experts'], (1, seq_len, cfg['top_k']), device=device)

            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            prefill_mean = np.mean(times)
            prefill_tok = seq_len / (prefill_mean / 1000)

            del h, w, ids
            torch.cuda.empty_cache()

            # Decode
            h = torch.randn(1, 1, cfg['hidden_dim'], dtype=torch.float16, device=device)
            w = torch.rand(1, 1, cfg['top_k'], dtype=torch.float16, device=device)
            w = w / w.sum(dim=-1, keepdim=True)
            ids = torch.randint(0, cfg['num_experts'], (1, 1, cfg['top_k']), device=device)

            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            decode_mean = np.mean(times)
            decode_tok = 1 / (decode_mean / 1000)

            del h, w, ids
            torch.cuda.empty_cache()

            results.append({
                'seq_len': seq_len,
                'prefill_tok_per_sec': float(prefill_tok),
                'decode_tok_per_sec': float(decode_tok),
                'status': 'OK',
            })

            print(f"{seq_len:<10} {prefill_tok:<20.0f} {decode_tok:<20.1f} OK")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"{seq_len:<10} {'-':<20} {'-':<20} OOM")
                results.append({'seq_len': seq_len, 'status': 'OOM'})
                break
            else:
                raise

    print()
    print(f'峰值内存: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GiB')

    # 保存结果
    output = {
        'config': cfg,
        'optimization': {'enable_cache': args.enable_cache, 'cache_hot': args.cache_hot},
        'results': results,
    }

    filename = f'large_context_quick_{"cache" if args.enable_cache else "baseline"}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'结果已保存: {filename}')


if __name__ == '__main__':
    main()
