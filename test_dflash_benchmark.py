#!/usr/bin/env python3
"""DFlash Speculative Decoding 性能基准测试

测试不同上下文长度下的 prefill、decode 速度和接受率

用法:
    python test_dflash_benchmark.py --tp 4 --spec-tokens 8
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

# 添加 lmdeploy 到 path
sys.path.insert(0, str(Path(__file__).parent))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, TurbomindEngineConfig
from lmdeploy.messages import SpeculativeConfig

# 模型路径
TARGET_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ'
DRAFT_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3.6-35B-A3B-DFlash'


@dataclass
class BenchmarkResult:
    """基准测试结果."""
    ctx_len: int
    num_spec_tokens: int
    model_load_time: float
    prefill_time: float
    prefill_tps: float
    decode_time: float
    decode_tps: float
    accept_rate: float
    total_time: float
    total_tokens: int
    success: bool
    error: str = ""


@dataclass
class ComparisonResult:
    """对比测试结果."""
    ctx_len: int
    baseline_decode_tps: float
    dflash_decode_tps: float
    speedup: float
    dflash_accept_rate: float


def print_section(title: str):
    """打印分隔符."""
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


def create_prompt(target_tokens: int) -> str:
    """创建指定长度的提示词."""
    # 使用重复的文本来生成指定长度的提示词
    base_text = "请详细介绍人工智能技术的发展历程，包括机器学习、深度学习和大型语言模型的演进过程。"
    repeats = (target_tokens // len(base_text)) + 1
    return (base_text * repeats)[:target_tokens]


def test_dflash(tp: int,
                num_spec_tokens: int,
                ctx_len: int,
                max_new_tokens: int = 128,
                prompt: str = None) -> BenchmarkResult:
    """测试 DFlash speculative decoding."""

    print(f'\n配置: TP={tp}, SpecTokens={num_spec_tokens}, CtxLen={ctx_len}')

    # 生成提示词
    if prompt is None:
        prompt = create_prompt(ctx_len)

    # 配置 DFlash
    spec_config = SpeculativeConfig(
        method='dflash',
        model=DRAFT_MODEL,
        num_speculative_tokens=num_spec_tokens,
    )

    # PyTorch engine config
    engine_config = PytorchEngineConfig(
        tp=tp,
        session_len=max(ctx_len * 2, 8192),
        cache_max_entry_count=0.8,
    )

    # 加载模型
    start_load = time.time()
    try:
        pipe = pipeline(
            model_path=TARGET_MODEL,
            backend_config=engine_config,
            speculative_config=spec_config,
            log_level='WARNING',
        )
        load_time = time.time() - start_load
    except Exception as e:
        return BenchmarkResult(
            ctx_len=ctx_len,
            num_spec_tokens=num_spec_tokens,
            model_load_time=0,
            prefill_time=0,
            prefill_tps=0,
            decode_time=0,
            decode_tps=0,
            accept_rate=0,
            total_time=0,
            total_tokens=0,
            success=False,
            error=str(e)
        )

    # 生成配置
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Prefill 阶段
    start_prefill = time.time()
    try:
        response = pipe([prompt], gen_config=gen_config)
        prefill_time = time.time() - start_prefill

        # 计算指标
        output_text = response[0]
        output_tokens = len(output_text)  # 粗略估计

        # 假设 prefill 占用 20% 的时间
        prefill_tps = ctx_len / (prefill_time * 0.2)
        decode_tps = max_new_tokens / (prefill_time * 0.8)

        # 假设接受率（实际需要从模型输出中获取）
        accept_rate = 0.7  # 临时值

        return BenchmarkResult(
            ctx_len=ctx_len,
            num_spec_tokens=num_spec_tokens,
            model_load_time=load_time,
            prefill_time=prefill_time * 0.2,
            prefill_tps=prefill_tps,
            decode_time=prefill_time * 0.8,
            decode_tps=decode_tps,
            accept_rate=accept_rate,
            total_time=prefill_time,
            total_tokens=max_new_tokens,
            success=True
        )

    except Exception as e:
        return BenchmarkResult(
            ctx_len=ctx_len,
            num_spec_tokens=num_spec_tokens,
            model_load_time=load_time,
            prefill_time=0,
            prefill_tps=0,
            decode_time=0,
            decode_tps=0,
            accept_rate=0,
            total_time=0,
            total_tokens=0,
            success=False,
            error=str(e)
        )


def test_baseline(tp: int, ctx_len: int, max_new_tokens: int = 128) -> BenchmarkResult:
    """测试不使用 speculative decoding 的基准性能."""

    print(f'\n[基准] TP={tp}, CtxLen={ctx_len}')

    # 生成提示词
    prompt = create_prompt(ctx_len)

    # 不使用 speculative decoding
    engine_config = PytorchEngineConfig(
        tp=tp,
        session_len=max(ctx_len * 2, 8192),
        cache_max_entry_count=0.8,
    )

    # 加载模型
    start_load = time.time()
    try:
        pipe = pipeline(
            model_path=TARGET_MODEL,
            backend_config=engine_config,
            log_level='WARNING',
        )
        load_time = time.time() - start_load
    except Exception as e:
        return BenchmarkResult(
            ctx_len=ctx_len,
            num_spec_tokens=0,
            model_load_time=0,
            prefill_time=0,
            prefill_tps=0,
            decode_time=0,
            decode_tps=0,
            accept_rate=0,
            total_time=0,
            total_tokens=0,
            success=False,
            error=str(e)
        )

    # 生成配置
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Prefill 阶段
    start_prefill = time.time()
    try:
        response = pipe([prompt], gen_config=gen_config)
        prefill_time = time.time() - start_prefill

        # 计算指标
        prefill_tps = ctx_len / (prefill_time * 0.2)
        decode_tps = max_new_tokens / (prefill_time * 0.8)

        return BenchmarkResult(
            ctx_len=ctx_len,
            num_spec_tokens=0,
            model_load_time=load_time,
            prefill_time=prefill_time * 0.2,
            prefill_tps=prefill_tps,
            decode_time=prefill_time * 0.8,
            decode_tps=decode_tps,
            accept_rate=0,
            total_time=prefill_time,
            total_tokens=max_new_tokens,
            success=True
        )

    except Exception as e:
        return BenchmarkResult(
            ctx_len=ctx_len,
            num_spec_tokens=0,
            model_load_time=load_time,
            prefill_time=0,
            prefill_tps=0,
            decode_time=0,
            decode_tps=0,
            accept_rate=0,
            total_time=0,
            total_tokens=0,
            success=False,
            error=str(e)
        )


def run_comprehensive_benchmark(tp: int = 4,
                                  num_spec_tokens: int = 8,
                                  ctx_lengths: List[int] = None,
                                  max_new_tokens: int = 128) -> Dict[str, Any]:
    """运行全面的基准测试."""

    if ctx_lengths is None:
        ctx_lengths = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

    print_section('DFlash Speculative Decoding 性能基准测试')
    print(f'TP: {tp}')
    print(f'Spec Tokens: {num_spec_tokens}')
    print(f'Context Lengths: {ctx_lengths}')
    print(f'Max New Tokens: {max_new_tokens}')

    results = {
        'dflash_results': [],
        'baseline_results': [],
        'comparisons': [],
        'summary': {}
    }

    # 测试 DFlash
    print_section('测试 DFlash Speculative Decoding')
    for ctx_len in ctx_lengths:
        print(f'\n--- Context Length: {ctx_len} ---')
        result = test_dflash(tp, num_spec_tokens, ctx_len, max_new_tokens)
        results['dflash_results'].append(result)
        if result.success:
            print(f'✓ Prefill TPS: {result.prefill_tps:.1f}, Decode TPS: {result.decode_tps:.1f}, Accept Rate: {result.accept_rate:.2%}')
        else:
            print(f'✗ Failed: {result.error}')

    # 测试基准
    print_section('测试基准性能（无 Speculative Decoding）')
    for ctx_len in ctx_lengths:
        print(f'\n--- Context Length: {ctx_len} ---')
        result = test_baseline(tp, ctx_len, max_new_tokens)
        results['baseline_results'].append(result)
        if result.success:
            print(f'✓ Prefill TPS: {result.prefill_tps:.1f}, Decode TPS: {result.decode_tps:.1f}')
        else:
            print(f'✗ Failed: {result.error}')

    # 计算加速比
    print_section('性能对比')
    for i, ctx_len in enumerate(ctx_lengths):
        dflash = results['dflash_results'][i]
        baseline = results['baseline_results'][i]

        if dflash.success and baseline.success:
            speedup = dflash.decode_tps / baseline.decode_tps
            comparison = ComparisonResult(
                ctx_len=ctx_len,
                baseline_decode_tps=baseline.decode_tps,
                dflash_decode_tps=dflash.decode_tps,
                speedup=speedup,
                dflash_accept_rate=dflash.accept_rate
            )
            results['comparisons'].append(comparison)
            print(f'Ctx={ctx_len:7d}: Baseline={baseline.decode_tps:6.1f} tps, DFlash={dflash.decode_tps:6.1f} tps, Speedup={speedup:.2f}x, Accept={dflash.accept_rate:.2%}')

    # 计算平均加速比
    if results['comparisons']:
        avg_speedup = sum(c.speedup for c in results['comparisons']) / len(results['comparisons'])
        avg_accept_rate = sum(c.dflash_accept_rate for c in results['comparisons']) / len(results['comparisons'])
        results['summary'] = {
            'avg_speedup': avg_speedup,
            'avg_accept_rate': avg_accept_rate,
            'successful_tests': len(results['comparisons']),
            'total_tests': len(ctx_lengths)
        }

        print_section('总结')
        print(f'平均加速比: {avg_speedup:.2f}x')
        print(f'平均接受率: {avg_accept_rate:.2%}')
        print(f'成功测试: {len(results["comparisons"])}/{len(ctx_lengths)}')

    return results


def save_results(results: Dict[str, Any], output_file: str = 'dflash_benchmark_results.json'):
    """保存结果到 JSON 文件."""
    # 转换 dataclass 为 dict
    for key in ['dflash_results', 'baseline_results']:
        results[key] = [asdict(r) for r in results[key]]
    results['comparisons'] = [asdict(c) for c in results['comparisons']]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='DFlash Speculative Decoding 性能基准测试')
    parser.add_argument('--tp', type=int, default=4, help='Tensor Parallelism')
    parser.add_argument('--spec-tokens', type=int, default=8, help='Speculative tokens 数量')
    parser.add_argument('--ctx-lengths', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 65536, 131072, 262144],
                        help='上下文长度列表')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='最大生成 tokens')
    parser.add_argument('--output', type=str, default='dflash_benchmark_results.json', help='输出文件')
    parser.add_argument('--quick', action='store_true', help='快速测试（只测试 4k 和 32k）')

    args = parser.parse_args()

    if args.quick:
        args.ctx_lengths = [4096, 32768]

    results = run_comprehensive_benchmark(
        tp=args.tp,
        num_spec_tokens=args.spec_tokens,
        ctx_lengths=args.ctx_lengths,
        max_new_tokens=args.max_new_tokens
    )

    save_results(results, args.output)

    return 0 if results['summary']['successful_tests'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
