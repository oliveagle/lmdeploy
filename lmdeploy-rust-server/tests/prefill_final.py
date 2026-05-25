#!/usr/bin/env python3
"""
准确的 Python prefill 性能测试
- 只测量第一个 token 的到达时间 (TTFT)
- 每次使用不同的 session_id 避免 KV cache 复用
- max_new_tokens=1 最小化 decode 时间
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count):
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats


async def main():
    print("=" * 80)
    print("Python TurboMind Prefill 性能测试（准确版）")
    print("=" * 80)
    print(f"Model: {MODEL}")

    tm = TurboMind(MODEL, engine_config=TurbomindEngineConfig(
        session_len=16384, max_batch_size=1, tp=1, model_format='awq',
        cache_max_entry_count=0.4, enable_prefix_caching=False,
    ))
    tok = Tokenizer(MODEL)

    test_cases = [("1K", 1024), ("4K", 4096), ("8K", 8192)]
    results = {}

    base_session = 50000

    for label, target_tokens in test_cases:
        prompt = gen_prompt(target_tokens)
        input_ids = tok.encode(prompt)
        actual_tokens = len(input_ids)
        print(f"\n测试 {label} ({actual_tokens} tokens)...")

        # Warmup
        warmup_ids = tok.encode(gen_prompt(512))
        for i in range(2):
            inst = tm.create_instance()
            async for _ in inst.async_stream_infer(
                session_id=base_session + i,
                input_ids=warmup_ids,
                gen_config=GenerationConfig(max_new_tokens=1),
                sequence_start=True, sequence_end=True,
            ):
                pass
        base_session += 10

        # 测量
        times = []
        for r in range(5):
            inst = tm.create_instance()
            session_id = base_session + r * 100

            start = time.perf_counter()
            async for out in inst.async_stream_infer(
                session_id=session_id,
                input_ids=input_ids,
                gen_config=GenerationConfig(max_new_tokens=1),
                sequence_start=True, sequence_end=True,
            ):
                if out.status.value in (1, 2):
                    first_token_ms = (time.perf_counter() - start) * 1000
                    times.append(first_token_ms)
                    print(f"  Run {r}: {first_token_ms:.1f}ms -> {actual_tokens / first_token_ms * 1000:.0f} tok/s")
                    break

        if times:
            avg_ms = sum(times) / len(times)
            tps = actual_tokens / avg_ms * 1000
            results[label] = {
                "tokens": actual_tokens,
                "times_ms": times,
                "avg_ms": avg_ms,
                "prefill_tps": tps,
            }
            print(f"  Avg: {avg_ms:.1f}ms -> {tps:.0f} tok/s")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"{'Context':>8} | {'Tokens':>8} | {'TTFT Avg':>10} | {'Prefill (tok/s)':>15}")
    print("-" * 60)
    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"{label:>8} | {r['tokens']:>8} | {r['avg_ms']:>8.1f}ms | {r['prefill_tps']:>13.0f}")

    # 历史对比
    print("\n历史数据 (Qwen3.5-9B):")
    for label in ["1K", "4K", "8K"]:
        hist = {"1K": {"ttft": 69.62, "tps": 14827}, "4K": {"ttft": 122.06, "tps": 33728}, "8K": {"ttft": 191.37, "tps": 42875}}[label]
        if label in results:
            cur = results[label]
            print(f"  {label}: 当前 TTFT={cur['avg_ms']:.1f}ms, TPS={cur['prefill_tps']:.0f} | 历史 TTFT={hist['ttft']:.2f}ms, TPS={hist['tps']:.0f}")

    output = {"engine": "LMDeploy Python TurboMind", "model": MODEL, "date": time.time(), "results": results}
    out_path = Path(__file__).parent / "prefill_accurate_python.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    tm.close()
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
