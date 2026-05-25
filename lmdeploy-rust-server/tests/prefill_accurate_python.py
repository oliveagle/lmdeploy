#!/usr/bin/env python3
"""
准确测量 Python TurboMind prefill 性能

关键发现：
- pipeline.stream_response=True 不返回有效的 gen_token_len
- TurboMind stream_output=True 返回 token-by-token 输出
- 历史 42K tok/s 是 Qwen3.5-9B，不是 Qwen3.6-35B-A3B-AWQ

测试方法：使用 TurboMind stream_output=True 准确测量 TTFT
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


async def measure_prefill(tm, input_ids, session_id, runs=3):
    """使用 TurboMind stream_output=True 测量 prefill 性能"""
    inst = tm.create_instance()

    times = []
    for r in range(runs):
        start = time.perf_counter()
        first_token_time = None
        async for output in inst.async_stream_infer(
            session_id=session_id * 1000 + r,
            input_ids=input_ids,
            gen_config=GenerationConfig(max_new_tokens=10, temperature=0.7, do_sample=False),
            sequence_start=True,
            sequence_end=True,
            stream_output=True,
        ):
            if first_token_time is None:
                first_token_time = (time.perf_counter() - start) * 1000
                break  # 只测量 TTFT

        if first_token_time:
            times.append(first_token_time)

    return times


async def main():
    print("=" * 80)
    print("Python TurboMind 准确 Prefill 性能测试")
    print("=" * 80)
    print(f"Model: {MODEL}")

    # 加载引擎
    print("\n[1/3] 加载 TurboMind...", end=" ", flush=True)
    tm = TurboMind(MODEL, engine_config=TurbomindEngineConfig(
        session_len=16384,
        max_batch_size=1,
        tp=1,
        model_format='awq',
        cache_max_entry_count=0.4,
        enable_prefix_caching=False,
    ))
    print("OK")

    tok = Tokenizer(MODEL)

    # 测试配置
    test_cases = [
        ("1K", 1024),
        ("4K", 4096),
        ("8K", 8192),
    ]

    results = {}

    for label, target_tokens in test_cases:
        print(f"\n[2/3] 测试 {label} context...")
        prompt = gen_prompt(target_tokens)
        input_ids = tok.encode(prompt)
        actual_tokens = len(input_ids)
        print(f"  Token count: {actual_tokens}")

        # Warmup
        warmup_ids = tok.encode(gen_prompt(512))
        print("  Warmup...", end=" ", flush=True)
        for i in range(2):
            inst = tm.create_instance()
            async for _ in inst.async_stream_infer(
                session_id=9000 + i,
                input_ids=warmup_ids,
                gen_config=GenerationConfig(max_new_tokens=5),
                sequence_start=True,
                sequence_end=True,
            ):
                pass
        print("OK")

        # 测量
        print(f"  Measure (3 runs)...", end=" ", flush=True)
        times = await measure_prefill(tm, input_ids, session_id=10 + test_cases.index((label, target_tokens)), runs=3)
        if times:
            avg_ms = sum(times) / len(times)
            tps = actual_tokens / avg_ms * 1000
            results[label] = {
                "context_tokens": actual_tokens,
                "ttft_ms": times,
                "ttft_avg_ms": avg_ms,
                "prefill_tps": tps,
            }
            print(f"TTFT: {[f'{t:.1f}ms' for t in times]}, avg: {avg_ms:.1f}ms, TPS: {tps:.0f}")
        else:
            print("Failed")

    # 总结
    print("\n" + "=" * 80)
    print("Python TurboMind Prefill 性能")
    print("=" * 80)
    print(f"{'Context':>8} | {'Tokens':>8} | {'TTFT (ms)':>12} | {'Prefill (tok/s)':>15}")
    print("-" * 60)

    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"{label:>8} | {r['context_tokens']:>8} | {r['ttft_avg_ms']:>10.1f}ms | {r['prefill_tps']:>13.0f}")

    # 对比历史数据
    print("\n" + "=" * 80)
    print("对比历史数据")
    print("=" * 80)

    historical_9b = {
        "1K": {"ttft": 69.62, "prefill": 14827.2},
        "4K": {"ttft": 122.06, "prefill": 33727.5},
        "8K": {"ttft": 191.37, "prefill": 42875.0},
    }

    historical_35b = {
        "1K": {"ttft": 1478.48, "prefill": 249.2},
        "4K": {"ttft": 1547.19, "prefill": 943.9},
        "8K": {"ttft": 1656.81, "prefill": 1760.2},
    }

    print(f"\n{'Context':>8} | {'Current TTFT':>12} | {'9B Historic':>12} | {'35B Historic':>14}")
    print("-" * 70)

    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            h9b = historical_9b[label]
            h35b = historical_35b[label]
            print(f"{label:>8} | {r['ttft_avg_ms']:>10.1f}ms | {h9b['ttft']:>7.2f}ms | {h35b['ttft']:>9.2f}ms")

    # 保存结果
    output = {
        "engine": "LMDeploy Python TurboMind (stream_output=True)",
        "model": MODEL,
        "date": time.time(),
        "results": results,
    }

    out_path = Path(__file__).parent / "prefill_accurate_python.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    tm.close()
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
