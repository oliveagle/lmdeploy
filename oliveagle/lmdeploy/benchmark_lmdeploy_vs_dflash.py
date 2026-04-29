#!/usr/bin/env python3
"""lmdeploy 基准测试 - 逐个测试

每次只运行一个测试，避免 OOM
"""

import sys
import os
import time
import gc
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import torch
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, PytorchEngineConfig
from transformers import AutoTokenizer

# 配置
TARGET_MODEL = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B"
DRAFT_MODEL = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash"
OUTPUT_TOKENS = 128
NUM_SPEC_TOKENS = 8

BASE_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20

# 测试配置
TESTS = [
    ('baseline', 4096),
    ('dflash', 4096),
    ('baseline', 8192),
    ('dflash', 8192),
    ('baseline', 16384),
    ('dflash', 16384),
]


def run_test(mode, ctx_len):
    """运行单个测试"""
    # 准备 prompt
    repeat_times = max(1, ctx_len // 300)
    prompt = BASE_PROMPT * repeat_times

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    prompt_tokens = len(tokenizer.encode(prompt))

    print(f"\n{'='*80}")
    print(f"测试: {mode}, Context: {ctx_len//1024}K ({prompt_tokens} tokens)")
    print(f"{'='*80}")

    # 清理 GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 创建 pipeline
    if mode == 'dflash':
        from lmdeploy.messages import SpeculativeConfig
        spec_config = SpeculativeConfig(
            method='dflash',
            model=DRAFT_MODEL,
            num_speculative_tokens=NUM_SPEC_TOKENS,
        )
        backend_config = PytorchEngineConfig(
            dtype='float16',
            cache_max_entry_count=0.5,
            eager_mode=True,
        )
    else:
        spec_config = None
        backend_config = TurbomindEngineConfig(
            dtype='float16',
            cache_max_entry_count=0.7,
        )

    pipe = pipeline(
        TARGET_MODEL,
        speculative_config=spec_config,
        backend_config=backend_config,
        log_level='WARNING',
    )

    # Prefill 测试
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    _ = pipe(prompt, GenerationConfig(max_new_tokens=1))
    torch.cuda.synchronize()
    prefill_time = time.time() - start
    prefill_throughput = prompt_tokens / prefill_time

    print(f"Prefill: {prefill_time*1000:.0f}ms, {prefill_throughput:.0f} t/s")

    # Decode 测试
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    output = pipe(
        prompt,
        GenerationConfig(
            max_new_tokens=OUTPUT_TOKENS,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        ),
    )
    torch.cuda.synchronize()
    decode_time = time.time() - start
    decode_throughput = OUTPUT_TOKENS / decode_time

    print(f"Decode: {decode_time*1000:.0f}ms, {decode_throughput:.1f} t/s")

    return {
        'mode': mode,
        'ctx_len': ctx_len,
        'prompt_tokens': prompt_tokens,
        'output_tokens': OUTPUT_TOKENS,
        'prefill_time_ms': round(prefill_time * 1000, 1),
        'prefill_throughput': round(prefill_throughput, 1),
        'decode_time_ms': round(decode_time * 1000, 1),
        'decode_throughput': round(decode_throughput, 1),
    }


def main():
    print("=" * 80)
    print("lmdeploy vs lmdeploy+DFlash 基准测试")
    print("=" * 80)
    print(f"\nTarget: {TARGET_MODEL}")
    print(f"Draft: {DRAFT_MODEL}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print(f"Speculative tokens: {NUM_SPEC_TOKENS}")

    results = []

    for mode, ctx_len in TESTS:
        try:
            result = run_test(mode, ctx_len)
            results.append(result)
        except Exception as e:
            print(f"错误: {e}")
            results.append({
                'mode': mode,
                'ctx_len': ctx_len,
                'error': str(e)[:100],
            })

        # 清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 打印结果汇总
    print(f"\n{'='*100}")
    print("结果汇总")
    print(f"{'='*100}")
    print(f"\n{'Context':^8} | {'Mode':^16} | {'Prefill (ms)':^12} | {'Prefill (t/s)':^12} | {'Decode (ms)':^12} | {'Decode (t/s)':^12}")
    print("-" * 100)

    baseline_by_ctx = {}
    dflash_by_ctx = {}

    for r in results:
        if 'error' not in r:
            if r['mode'] == 'baseline':
                baseline_by_ctx[r['ctx_len']] = r
            else:
                dflash_by_ctx[r['ctx_len']] = r

    for ctx_len in [4096, 8192, 16384]:
        ctx_label = f"{ctx_len//1024}K"
        baseline = baseline_by_ctx.get(ctx_len)
        dflash = dflash_by_ctx.get(ctx_len)

        if baseline:
            print(f"{ctx_label:^8} | {'lmdeploy':^16} | {baseline['prefill_time_ms']:^12.0f} | {baseline['prefill_throughput']:^12.0f} | {baseline['decode_time_ms']:^12.0f} | {baseline['decode_throughput']:^12.1f}")
        else:
            print(f"{ctx_label:^8} | {'lmdeploy':^16} | {'N/A':^12} | {'N/A':^12} | {'N/A':^12} | {'N/A':^12}")

        if dflash:
            speedup = ''
            if baseline:
                speedup = f"{baseline['decode_throughput']/dflash['decode_throughput']:.2f}x" if dflash['decode_throughput'] > 0 else ''
            print(f"{'':^8} | {'lmdeploy+DFlash':^16} | {dflash['prefill_time_ms']:^12.0f} | {dflash['prefill_throughput']:^12.0f} | {dflash['decode_time_ms']:^12.0f} | {dflash['decode_throughput']:^12.1f} {speedup}")
        else:
            print(f"{'':^8} | {'lmdeploy+DFlash':^16} | {'N/A':^12} | {'N/A':^12} | {'N/A':^12} | {'N/A':^12}")

    print(f"{'='*100}")


if __name__ == '__main__':
    main()