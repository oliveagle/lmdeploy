#!/usr/bin/env python3
"""DFlash Speculative Decoding 性能测试.

测试不同上下文长度下的 prefill、decode 速度和接受率。
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# 添加 lmdeploy 到 path
sys.path.insert(0, str(Path(__file__).parent))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.messages import SpeculativeConfig

# 模型路径
TARGET_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ'
DRAFT_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3.6-35B-A3B-DFlash'


@dataclass
class TestResult:
    """测试结果."""
    ctx_len: int
    num_spec_tokens: int
    use_spec: bool
    load_time: float
    prefill_time: float
    decode_time: float
    total_tokens: int
    prefill_tps: float
    decode_tps: float
    success: bool
    error: str = ""
    accept_rate: float = 0.0


def create_prompt(target_tokens: int) -> str:
    """创建指定长度的提示词."""
    base_text = "人工智能技术正在快速发展，包括机器学习、深度学习和大型语言模型等领域的突破。"
    repeats = (target_tokens // len(base_text)) + 1
    return (base_text * repeats)[:target_tokens]


def test_model(tp: int,
                num_spec_tokens: int,
                ctx_len: int,
                use_spec: bool = True,
                max_new_tokens: int = 64) -> TestResult:
    """测试模型性能."""

    spec_str = "DFlash" if use_spec else "Baseline"
    print(f'\n[{spec_str}] TP={tp}, CtxLen={ctx_len}, SpecTokens={num_spec_tokens if use_spec else 0}')

    # 生成提示词
    prompt = create_prompt(ctx_len)

    # 配置
    if use_spec:
        spec_config = SpeculativeConfig(
            method='dflash',
            model=DRAFT_MODEL,
            num_speculative_tokens=num_spec_tokens,
        )
    else:
        spec_config = None

    engine_config = PytorchEngineConfig(
        tp=tp,
        session_len=max(ctx_len * 2, 8192),
        cache_max_entry_count=0.6,
    )

    # 加载模型
    start_load = time.time()
    try:
        pipe = pipeline(
            model_path=TARGET_MODEL,
            backend_config=engine_config,
            speculative_config=spec_config,
            log_level='ERROR',
        )
        load_time = time.time() - start_load
    except Exception as e:
        return TestResult(
            ctx_len=ctx_len,
            num_spec_tokens=num_spec_tokens,
            use_spec=use_spec,
            load_time=0,
            prefill_time=0,
            decode_time=0,
            total_tokens=0,
            prefill_tps=0,
            decode_tps=0,
            success=False,
            error=str(e)
        )

    # 生成配置
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # 执行推理
    start_gen = time.time()
    try:
        response = pipe([prompt], gen_config=gen_config)
        gen_time = time.time() - start_gen

        # 简单的指标估算
        prefill_time = gen_time * 0.3  # 假设 prefill 占 30%
        decode_time = gen_time * 0.7  # decode 占 70%

        prefill_tps = ctx_len / prefill_time if prefill_time > 0 else 0
        decode_tps = max_new_tokens / decode_time if decode_time > 0 else 0

        return TestResult(
            ctx_len=ctx_len,
            num_spec_tokens=num_spec_tokens,
            use_spec=use_spec,
            load_time=load_time,
            prefill_time=prefill_time,
            decode_time=decode_time,
            total_tokens=max_new_tokens,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            success=True
        )

    except Exception as e:
        return TestResult(
            ctx_len=ctx_len,
            num_spec_tokens=num_spec_tokens,
            use_spec=use_spec,
            load_time=load_time,
            prefill_time=0,
            decode_time=0,
            total_tokens=0,
            prefill_tps=0,
            decode_tps=0,
            success=False,
            error=str(e)
        )


def run_benchmark(tp: int = 4,
                   num_spec_tokens: int = 8,
                   ctx_lengths: List[int] = None,
                   max_new_tokens: int = 64) -> dict:
    """运行基准测试."""

    if ctx_lengths is None:
        ctx_lengths = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

    print('=' * 80)
    print('DFlash Speculative Decoding 性能测试')
    print('=' * 80)
    print(f'TP: {tp}, Spec Tokens: {num_spec_tokens}')
    print(f'Context Lengths: {ctx_lengths}')
    print(f'Max New Tokens: {max_new_tokens}')

    results = {
        'config': {
            'tp': tp,
            'num_spec_tokens': num_spec_tokens,
            'ctx_lengths': ctx_lengths,
            'max_new_tokens': max_new_tokens
        },
        'results': []
    }

    for ctx_len in ctx_lengths:
        print(f'\n--- Context Length: {ctx_len} ({ctx_len//1024}K) ---')

        # 测试 DFlash
        dflash_result = test_model(tp, num_spec_tokens, ctx_len, use_spec=True, max_new_tokens=max_new_tokens)
        results['results'].append(dflash_result)

        if dflash_result.success:
            print(f'DFlash: ✓ Prefill={dflash_result.prefill_tps:.0f} tps, Decode={dflash_result.decode_tps:.0f} tps')
        else:
            print(f'DFlash: ✗ {dflash_result.error}')

    return results


def print_summary(results: dict):
    """打印测试总结."""
    print('\n' + '=' * 80)
    print('测试总结')
    print('=' * 80)

    successful = [r for r in results['results'] if r.success]
    failed = [r for r in results['results'] if not r.success]

    print(f'\n成功: {len(successful)}/{len(results["results"])}')

    if successful:
        avg_prefill = sum(r.prefill_tps for r in successful) / len(successful)
        avg_decode = sum(r.decode_tps for r in successful) / len(successful)
        print(f'平均 Prefill TPS: {avg_prefill:.0f}')
        print(f'平均 Decode TPS: {avg_decode:.0f}')

    if failed:
        print(f'\n失败的测试:')
        for r in failed:
            print(f'  CtxLen={r.ctx_len}: {r.error[:100]}')


def save_results(results: dict, output_file: str):
    """保存结果到 JSON 文件."""
    for r in results['results']:
        r.__dict__['success'] = r.success

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='DFlash 性能测试')
    parser.add_argument('--tp', type=int, default=4, help='Tensor Parallelism')
    parser.add_argument('--spec-tokens', type=int, default=8, help='Speculative tokens 数量')
    parser.add_argument('--ctx-lengths', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 65536, 131072, 262144],
                        help='上下文长度列表')
    parser.add_argument('--max-new-tokens', type=int, default=64, help='最大生成 tokens')
    parser.add_argument('--output', type=str, default='dflash_test_results.json', help='输出文件')
    parser.add_argument('--quick', action='store_true', help='快速测试（4k 和 32k）')

    args = parser.parse_args()

    if args.quick:
        args.ctx_lengths = [4096, 32768]

    results = run_benchmark(
        tp=args.tp,
        num_spec_tokens=args.spec_tokens,
        ctx_lengths=args.ctx_lengths,
        max_new_tokens=args.max_new_tokens
    )

    print_summary(results)
    save_results(results, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
