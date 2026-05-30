#!/usr/bin/env python3
"""
LMDeploy TurboMind Benchmark Script
Direct engine path, no HTTP overhead.
"""
import asyncio
import json
import time
from datetime import datetime

from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig
import numpy as np

MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

# Test scenarios - match the archive
SCENARIOS = [
    {"name": "short_context", "input_len": 512, "output_len": 512},
    {"name": "medium_context", "input_len": 1024, "output_len": 512},
    {"name": "long_context", "input_len": 4096, "output_len": 512},
    {"name": "long_context", "input_len": 8192, "output_len": 512},
]

NUM_REQUESTS = 5  # Match archive
WARMUP_REQUESTS = 3


def generate_prompt(tokenizer, target_len):
    """Generate specified token length prompt"""
    base_tokens = tokenizer.encode("Hello, how are you today?")
    repeat_times = (target_len // len(base_tokens)) + 1
    input_ids = (base_tokens * repeat_times)[:target_len]
    return tokenizer.decode(input_ids), input_ids


async def run_single_benchmark(model_inst, input_ids, output_len, session_id):
    """Run single benchmark with detailed metrics"""
    params = GenerationConfig(
        max_new_tokens=output_len,
        ignore_eos=True,
    )

    start = time.perf_counter()
    ttft = None
    token_count = 0
    prefill_time = None

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

            if token_count == 0:
                # First token time = TTFT = prefill time
                prefill_time = now - start
                ttft = now - start

            token_count += len(outputs.token_ids)

        await gen.aclose()

        total_time = time.perf_counter() - start

        return {
            "ttft": ttft,
            "prefill_time": prefill_time,
            "total_time": total_time,
            "token_count": token_count,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None


async def benchmark_scenario(tm_model, tokenizer, scenario):
    """Benchmark single scenario"""
    input_len = scenario["input_len"]
    output_len = scenario["output_len"]

    _, input_ids = generate_prompt(tokenizer, input_len)
    actual_input_len = len(input_ids)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['name']}")
    print(f"Input length: {actual_input_len}, Output length: {output_len}")
    print(f"{'='*60}")

    results = []

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} requests...")
    model_inst = tm_model.create_instance()
    for i in range(WARMUP_REQUESTS):
        await run_single_benchmark(model_inst, [1,2,3,4,5], 10, i)

    # Actual benchmark - single concurrency to match archive
    print(f"Benchmark: {NUM_REQUESTS} requests...")
    for i in range(NUM_REQUESTS):
        result = await run_single_benchmark(model_inst, input_ids, output_len, WARMUP_REQUESTS + i)
        if result:
            results.append(result)
            print(f"  Run {i+1}: TTFT={result['ttft']*1000:.1f}ms, Tokens={result['token_count']}")

    if not results:
        print("No valid results!")
        return None

    # Statistics matching the official profiler methodology
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    total_times = [r["total_time"] for r in results]
    decode_times = [r["total_time"] - r["ttft"] for r in results if r["ttft"] is not None]
    avg_output_tokens = sum(r["token_count"] for r in results) / len(results)

    avg_ttft = np.mean(ttfts) if ttfts else float('inf')
    avg_total_time = np.mean(total_times)
    avg_decode_time = np.mean(decode_times) if decode_times else float('inf')

    # Prefill throughput: input_tokens / ttft (matches official formula)
    prefill_tps = actual_input_len / avg_ttft if avg_ttft > 0 else 0

    # Decode throughput: output_tokens / decode_time (matches official formula)
    decode_tps = avg_output_tokens / avg_decode_time if avg_decode_time > 0 else 0

    stats = {
        "scenario": scenario["name"],
        "input_len": actual_input_len,
        "output_len": output_len,
        "completed": len(results),
        "ttft_ms_avg": avg_ttft * 1000,
        "total_time_ms_avg": avg_total_time * 1000,
        "decode_time_ms_avg": avg_decode_time * 1000,
        "prefill_throughput_tps": prefill_tps,
        "decode_throughput_tps": decode_tps,
        "avg_output_tokens": avg_output_tokens,
    }

    print(f"\nResults:")
    print(f"  Completed: {stats['completed']}/{NUM_REQUESTS}")
    print(f"  TTFT avg: {stats['ttft_ms_avg']:.1f}ms")
    print(f"  Prefill throughput: {stats['prefill_throughput_tps']:.1f} tok/s")
    print(f"  Decode throughput: {stats['decode_throughput_tps']:.1f} tok/s")

    return stats


async def main():
    print(f"{'='*60}")
    print(f"LMDeploy TurboMind Benchmark - Direct Engine")
    print(f"Model: {MODEL_PATH}")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    print("Initializing TurboMind engine...")
    tm_model = TurboMind.from_pretrained(MODEL_PATH)

    all_results = []

    for scenario in SCENARIOS:
        result = await benchmark_scenario(tm_model, tokenizer, scenario)
        if result:
            all_results.append(result)
        await asyncio.sleep(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_turbomind_{timestamp}.json"

    output = {
        "engine": "LMDeploy TurboMind (direct)",
        "model": MODEL_PATH,
        "backend": "turbomind",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    print(f"\n{'='*80}")
    print("SUMMARY (vs Archive)")
    print(f"{'='*80}")
    print(f"{'Scenario':30s} | {'TTFT (ms)':>10s} | {'Arch TTFT':>10s} | {'Prefill':>12s} | {'Arch Prefill':>12s} | {'Decode':>10s} | {'Arch Decode':>10s}")
    print(f"{'-'*80}")

    # Archive expected values
    expected = {
        512: {"ttft": 80, "prefill": 6408, "decode": 42.6},
        1024: {"ttft": 139, "prefill": 7356, "decode": 42.2},
        4096: {"ttft": 402, "prefill": 10190, "decode": 40.5},
        8192: {"ttft": 649, "prefill": 12614, "decode": 39.8},
    }

    for r in all_results:
        e = expected.get(r["input_len"], {"ttft": 0, "prefill": 0, "decode": 0})
        inp_len = r["input_len"]
        print(f"{r['scenario']}({inp_len}):30s | {r['ttft_ms_avg']:10.1f} | {e['ttft']:10.1f} | {r['prefill_throughput_tps']:12.1f} | {e['prefill']:12.1f} | {r['decode_throughput_tps']:10.1f} | {e['decode']:10.1f}")


if __name__ == "__main__":
    asyncio.run(main())
