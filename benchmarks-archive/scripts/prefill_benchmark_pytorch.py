#!/usr/bin/env python3
"""
使用 PyTorch backend 测量 prefill 性能
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path

os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig

# 配置
MODEL = os.environ.get(
    "MODEL_PATH",
    "/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
)
WARMUP_RUNS = 2
MEASURE_RUNS = 5

# 测试 prompts
TEST_PROMPTS = {
    "256": "The quick brown fox jumps over the lazy dog. " * 40,
    "512": "The quick brown fox jumps over the lazy dog. " * 80,
    "1024": "The quick brown fox jumps over the lazy dog. " * 160,
    "2048": "The quick brown fox jumps over the lazy dog. " * 320,
    "4096": "The quick brown fox jumps over the lazy dog. " * 640,
}


async def measure_prefill(pipe, prompt: str, runs: int = MEASURE_RUNS):
    """测量 prefill 性能"""
    gen_cfg = GenerationConfig(max_new_tokens=1, temperature=0.7)
    run_times = []

    for r in range(runs):
        start = time.perf_counter()
        outputs = await pipe.async_infer([prompt], gen_config=gen_cfg)
        elapsed = (time.perf_counter() - start) * 1000
        run_times.append(elapsed)

    avg_ms = sum(run_times) / len(run_times)
    min_ms = min(run_times)
    max_ms = max(run_times)

    # 估算 token 数 (大约 4 chars/token)
    ctx_tokens = len(prompt) // 4
    prefill_tps_avg = (ctx_tokens / avg_ms * 1000) if avg_ms > 0 else 0
    prefill_tps_max = (ctx_tokens / min_ms * 1000) if min_ms > 0 else 0

    return {
        "ctx_tokens": ctx_tokens,
        "runs_ms": run_times,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "prefill_tps_avg": prefill_tps_avg,
        "prefill_tps_max": prefill_tps_max,
    }


async def main():
    print("=" * 80)
    print("PyTorch Backend Prefill 性能测试")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    print(f"Warmup: {WARMUP_RUNS}, Measure: {MEASURE_RUNS}")

    print("\n创建 pipeline...", end=" ", flush=True)
    pipe = pipeline(
        MODEL,
        backend_config=PytorchEngineConfig(
            session_len=8192 * 2,
            max_batch_size=32,
        ),
    )
    print("OK")

    print(f"Warmup ({WARMUP_RUNS} runs)...", end=" ", flush=True)
    for _ in range(WARMUP_RUNS):
        await pipe.async_infer(["Hello"], gen_config=GenerationConfig(max_new_tokens=1))
    print("OK")

    print("\n测量 prefill 性能...")
    results = {}

    print(f"\n{'Context':>8} | {'Avg TTFT':>10} | {'Min TTFT':>10} | "
          f"{'Avg TPS':>12} | {'Max TPS':>12}")
    print("-" * 70)

    for ctx_label, prompt in TEST_PROMPTS.items():
        result = await measure_prefill(pipe, prompt, MEASURE_RUNS)
        results[ctx_label] = result

        print(f"{result['ctx_tokens']:>8} | "
              f"{result['avg_ms']:>10.2f} | "
              f"{result['min_ms']:>10.2f} | "
              f"{result['prefill_tps_avg']:>12.0f} | "
              f"{result['prefill_tps_max']:>12.0f}")

    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    for ctx_label in ["256", "512", "1024", "2048", "4096"]:
        if ctx_label in results:
            r = results[ctx_label]
            print(f"\n  {ctx_label} tokens:")
            print(f"    Avg Prefill: {r['prefill_tps_avg']:.0f} tok/s (TTFT: {r['avg_ms']:.2f}ms)")
            print(f"    Max Prefill: {r['prefill_tps_max']:.0f} tok/s (TTFT: {r['min_ms']:.2f}ms)")

    out_path = Path(__file__).parent / "prefill_benchmark_pytorch.json"
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "results": results}, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
