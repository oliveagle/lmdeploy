#!/usr/bin/env python3
"""lmdeploy baseline 基准测试 - 4K, 8K, 16K context"""

import sys
import os
import time
import gc
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import torch
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from transformers import AutoTokenizer

# 配置
TARGET_MODEL = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B"
OUTPUT_TOKENS = 128
BASE_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20  # ~300 tokens

# 测试配置
TESTS = [4096, 8192, 16384]


def run_test(ctx_len):
    """运行单个测试"""
    repeat_times = max(1, ctx_len // 300)
    prompt = BASE_PROMPT * repeat_times

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    prompt_tokens = len(tokenizer.encode(prompt))

    print(f"\n{'='*80}")
    print(f"Context: {ctx_len//1024}K ({prompt_tokens} tokens)")
    print(f"{'='*80}")

    gc.collect()
    torch.cuda.empty_cache()

    pipe = pipeline(
        TARGET_MODEL,
        backend_config=TurbomindEngineConfig(
            dtype='float16',
            cache_max_entry_count=0.7,
        ),
        log_level='WARNING',
    )

    # Prefill
    torch.cuda.synchronize()
    start = time.time()
    _ = pipe(prompt, GenerationConfig(max_new_tokens=1))
    torch.cuda.synchronize()
    prefill_time = time.time() - start
    prefill_throughput = prompt_tokens / prefill_time

    print(f"Prefill: {prefill_time*1000:.0f}ms, {prefill_throughput:.0f} t/s")

    # Decode
    gc.collect()
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
        'ctx_len': ctx_len,
        'prompt_tokens': prompt_tokens,
        'prefill_time_ms': round(prefill_time * 1000, 1),
        'prefill_throughput': round(prefill_throughput, 1),
        'decode_time_ms': round(decode_time * 1000, 1),
        'decode_throughput': round(decode_throughput, 1),
    }


def main():
    print("=" * 80)
    print("lmdeploy 基准测试 - V100 SM_70")
    print("=" * 80)
    print(f"\nTarget: {TARGET_MODEL}")
    print(f"Output tokens: {OUTPUT_TOKENS}")

    results = []

    for ctx_len in TESTS:
        try:
            result = run_test(ctx_len)
            results.append(result)
        except Exception as e:
            print(f"错误: {e}")
            results.append({
                'ctx_len': ctx_len,
                'error': str(e)[:100],
            })

    # 汇总
    print(f"\n{'='*100}")
    print("结果汇总")
    print(f"{'='*100}")
    print(f"\n{'Context':^8} | {'Prompt Tokens':^12} | {'Prefill (ms)':^12} | {'Prefill (t/s)':^12} | {'Decode (ms)':^12} | {'Decode (t/s)':^12}")
    print("-" * 100)

    for r in results:
        if 'error' not in r:
            ctx_label = f"{r['ctx_len']//1024}K"
            print(f"{ctx_label:^8} | {r['prompt_tokens']:^12} | {r['prefill_time_ms']:^12.0f} | {r['prefill_throughput']:^12.0f} | {r['decode_time_ms']:^12.0f} | {r['decode_throughput']:^12.1f}")
        else:
            ctx_label = f"{r['ctx_len']//1024}K"
            print(f"{ctx_label:^8} | {'N/A':^12} | {'N/A':^12} | {'N/A':^12} | {'N/A':^12} | {'N/A':^12}")

    print(f"{'='*100}")


if __name__ == '__main__':
    main()