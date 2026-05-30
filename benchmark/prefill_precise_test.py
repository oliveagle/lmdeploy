#!/usr/bin/env python3
"""
精确 Prefill 对比测试
直接测量 TTFT 和 prefill throughput，排除 decode 和其他开销
"""
import asyncio
import json
import time
from datetime import datetime
import numpy as np

from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig

MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

# Test scenarios - prefill only
SCENARIOS = [
    {"name": "512", "input_len": 512},
    {"name": "1024", "input_len": 1024},
    {"name": "2048", "input_len": 2048},
    {"name": "4096", "input_len": 4096},
    {"name": "8192", "input_len": 8192},
]

NUM_REQUESTS = 10
WARMUP_REQUESTS = 3


def generate_prompt(tokenizer, target_len):
    """Generate specified token length prompt"""
    base_tokens = tokenizer.encode("Hello, how are you today? I'm doing great and excited to test this model.")
    repeat_times = (target_len // len(base_tokens)) + 1
    input_ids = (base_tokens * repeat_times)[:target_len]
    return input_ids


async def run_prefill_only(model_inst, input_ids, session_id):
    """Run prefill-only request (output_len=1, ignore_eos=True)"""
    params = GenerationConfig(
        max_new_tokens=1,
        ignore_eos=True,
    )

    start = time.perf_counter()
    ttft = None
    token_count = 0

    try:
        gen = model_inst.async_stream_infer(
            session_id,
            input_ids=input_ids,
            gen_config=params,
            sequence_start=True,
            sequence_end=True,
            stream_output=True,
        )
        async for outputs in gen:
            now = time.perf_counter()
            if ttft is None:
                ttft = now - start
            token_count += len(outputs.token_ids)

        await gen.aclose()
        return {"ttft": ttft, "token_count": token_count}

    except Exception as e:
        return None


async def benchmark_scenario(tm_model, tokenizer, scenario):
    """Benchmark single scenario"""
    input_len = scenario["input_len"]

    input_ids = generate_prompt(tokenizer, input_len)
    actual_input_len = len(input_ids)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['name']}, Input length: {actual_input_len}")
    print(f"{'='*60}")

    model_inst = tm_model.create_instance()

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} requests...")
    for i in range(WARMUP_REQUESTS):
        await run_prefill_only(model_inst, [1, 2, 3, 4, 5] * 20, i)
    await asyncio.sleep(0.5)

    # Benchmark
    print(f"Benchmark: {NUM_REQUESTS} requests...")
    results = []
    for i in range(NUM_REQUESTS):
        result = await run_prefill_only(model_inst, input_ids, WARMUP_REQUESTS + i)
        if result and result["ttft"]:
            ttft_ms = result["ttft"] * 1000
            prefill_tps = len(input_ids) / result["ttft"]
            results.append(result)
            print(f"  Run {i+1}: TTFT={ttft_ms:.1f}ms, Prefill={prefill_tps:.1f} tok/s")

    if not results:
        return None

    ttfts = [r["ttft"] for r in results]
    avg_ttft = np.mean(ttfts)
    std_ttft = np.std(ttfts)
    min_ttft = np.min(ttfts)
    max_ttft = np.max(ttfts)

    prefill_tps_list = [actual_input_len / t for t in ttfts]
    avg_prefill_tps = np.mean(prefill_tps_list)
    min_prefill_tps = np.min(prefill_tps_list)
    max_prefill_tps = np.max(prefill_tps_list)

    stats = {
        "scenario": scenario["name"],
        "input_len": actual_input_len,
        "completed": len(results),
        "ttft_ms_avg": avg_ttft * 1000,
        "ttft_ms_std": std_ttft * 1000,
        "ttft_ms_min": min_ttft * 1000,
        "ttft_ms_max": max_ttft * 1000,
        "prefill_tps_avg": avg_prefill_tps,
        "prefill_tps_min": min_prefill_tps,
        "prefill_tps_max": max_prefill_tps,
        "all_ttfts_ms": [t * 1000 for t in ttfts],
    }

    print(f"\nResults:")
    print(f"  TTFT: avg={stats['ttft_ms_avg']:.1f}ms, std={stats['ttft_ms_std']:.1f}ms, min={stats['ttft_ms_min']:.1f}ms, max={stats['ttft_ms_max']:.1f}ms")
    print(f"  Prefill: avg={stats['prefill_tps_avg']:.1f} tok/s, min={stats['prefill_tps_min']:.1f} tok/s, max={stats['prefill_tps_max']:.1f} tok/s")

    return stats


async def main():
    print(f"{'='*60}")
    print(f"Python TurboMind Prefill Benchmark (Precise)")
    print(f"Model: {MODEL_PATH}")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    print("Initializing TurboMind engine...")
    tm_model = TurboMind.from_pretrained(MODEL_PATH)
    print("Warmup completed\n")

    all_results = []

    for scenario in SCENARIOS:
        result = await benchmark_scenario(tm_model, tokenizer, scenario)
        if result:
            all_results.append(result)
        await asyncio.sleep(0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_prefill_python_precise_{timestamp}.json"

    output = {
        "engine": "Python TurboMind",
        "model": MODEL_PATH,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Input':>8s} | {'TTFT avg(ms)':>12s} | {'TTFT std':>10s} | {'Prefill avg':>12s} | {'Prefill max':>12s}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['input_len']:8d} | {r['ttft_ms_avg']:12.1f} | {r['ttft_ms_std']:10.1f} | {r['prefill_tps_avg']:12.1f} | {r['prefill_tps_max']:12.1f}")


if __name__ == "__main__":
    asyncio.run(main())