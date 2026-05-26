#!/usr/bin/env python3
"""
最终确认的 Python prefill 性能基准测试
使用真实的 token count，准确测量 TTFT
"""

import os
import sys
import time
import json
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent / "lmdeploy/lib"))

from lmdeploy import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count):
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats


async def main():
    print("=" * 80)
    print("Python TurboMind Prefill 准确性能测试")
    print("=" * 80)
    print()
    print("Model:", MODEL)

    tm = TurboMind(model_path=MODEL, engine_config=TurbomindEngineConfig(
        session_len=16384,
        max_batch_size=1,
        tp=1,
        model_format="awq",
        cache_max_entry_count=0.4
    ))
    tok = Tokenizer(MODEL)

    test_cases = [("1K", 1024), ("4K", 4096), ("8K", 8192)]

    results = {}

    # Warmup
    print()
    print("Warmup (2 runs)...")
    warmup_prompt = gen_prompt(512)
    warmup_input_ids = tok.encode(warmup_prompt)
    for i in range(2):
        async for _ in tm.create_instance().async_stream_infer(
            session_id=i, input_ids=warmup_input_ids,
            gen_config=GenerationConfig(max_new_tokens=1),
            sequence_start=True, sequence_end=True
        ):
            pass

    print()
    print("{:>8} | {:>10} | {:>12} | {:>12} | {:>15}".format(
        "Context", "Tokens", "TTFT Avg", "TTFT Min", "Prefill (tok/s)"))
    print("-" * 70)

    for label, target_tokens in test_cases:
        prompt = gen_prompt(target_tokens)
        input_ids = tok.encode(prompt)
        actual_tokens = len(input_ids)

        times = []
        for r in range(5):
            start = time.perf_counter()
            async for out in tm.create_instance().async_stream_infer(
                session_id=100 + r, input_ids=input_ids,
                gen_config=GenerationConfig(max_new_tokens=1),
                sequence_start=True, sequence_end=True
            ):
                if out.status.value in (1, 2):
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
                    print(f"{label} Run {r}: {elapsed:.1f}ms")
                    break

        if times:
            avg_ms = sum(times) / len(times)
            min_ms = min(times)
            tps = actual_tokens / avg_ms * 1000
            results[label] = {"tokens": actual_tokens, "ttft_avg_ms": avg_ms, "ttft_min_ms": min_ms, "tps": tps}
            print(f"{label} AVG: {avg_ms:.1f}ms ({tps:.0f} tok/s, min {min_ms:.1f}ms)")

    tm.close()

    print()
    print("=" * 80)
    print("总结 - Python TurboMind Prefill 性能")
    print("=" * 80)
    for label in ["1K", "4K", "8K"]:
        if label in results:
            print(f"  {label:>6}: {results[label]['tps']:>8.0f} tok/s (TTFT: {results[label]['ttft_avg_ms']:>7.1f}ms)")

    output = {
        "model": MODEL,
        "timestamp": time.time(),
        "results": results,
    }
    with open("/mnt/data/lmdeploy/lmdeploy-rust-server/tests/prefill_accurate_python.json", "w") as f:
        json.dump(output, f, indent=2)
    print()
    print("结果已保存: prefill_accurate_python.json")


import asyncio
asyncio.run(main())
