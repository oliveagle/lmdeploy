#!/usr/bin/env python3
"""
Comprehensive Python TurboMind Prefill Benchmark
Test multiple input lengths to get real prefill performance data
"""
import asyncio
import json
import time
from datetime import datetime
import numpy as np

from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig, GenerationConfig

MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

SCENARIOS = [
    {"name": "512", "input_len": 512},
    {"name": "1024", "input_len": 1024},
    {"name": "2048", "input_len": 2048},
    {"name": "4096", "input_len": 4096},
    {"name": "8192", "input_len": 8192},
]

NUM_REQUESTS = 10  # More requests for stable average
WARMUP = 3

def generate_prompt_ids(target_len):
    """Generate repeated token IDs for a target length"""
    base_ids = [12800, 12801, 12802, 12803, 12804, 12805, 12806, 12807, 12808, 12809]
    repeat = (target_len // len(base_ids)) + 1
    return (base_ids * repeat)[:target_len]


async def run_benchmark():
    print("=" * 80)
    print("Python TurboMind Prefill Benchmark (Comprehensive)")
    print(f"Model: {MODEL_PATH}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize engine
    engine_config = TurbomindEngineConfig(
        session_len=154880,
        max_batch_size=32,
        cache_max_entry_count=0.5,
        max_prefill_token_num=8192,
        quant_policy=4,  # AWQ
        enable_prefix_caching=False,
    )

    print("\nLoading model...")
    tm_model = TurboMind(MODEL_PATH, engine_config=engine_config)
    model_inst = tm_model.create_instance()
    print("Model loaded\n")

    all_results = {}

    for scenario in SCENARIOS:
        name = scenario["name"]
        input_len = scenario["input_len"]
        input_ids = generate_prompt_ids(input_len)
        actual_len = len(input_ids)

        print(f"\n{'='*60}")
        print(f"Scenario: {name} ({actual_len} tokens)")
        print(f"{'='*60}")

        # Warmup
        print(f"Warmup: {WARMUP} requests...")
        warmup_session = 0
        for i in range(WARMUP):
            gen = model_inst.async_stream_infer(
                warmup_session,
                input_ids=[1, 2, 3],
                gen_config=GenerationConfig(max_new_tokens=1, ignore_eos=True),
                stream_output=True,
                sequence_start=True,
                sequence_end=True,
            )
            async for _ in gen:
                pass
            await gen.aclose()
            warmup_session += 1

        time.sleep(0.5)

        # Benchmark
        print(f"Benchmark: {NUM_REQUESTS} requests...")
        ttfts = []
        base_session_id = 10000
        for i in range(NUM_REQUESTS):
            session_id = base_session_id + i
            start = time.perf_counter()
            first_token = True

            gen = model_inst.async_stream_infer(
                session_id,
                input_ids=input_ids,
                gen_config=GenerationConfig(max_new_tokens=1, ignore_eos=True),
                stream_output=True,
                sequence_start=True,
                sequence_end=True,
            )
            async for outputs in gen:
                if first_token:
                    ttft = time.perf_counter() - start
                    ttfts.append(ttft)
                    first_token = False
                    break  # Only measure first token
            await gen.aclose()

            if ttfts:  # Only print if we got a valid measurement
                print(f"  Run {i+1}: TTFT={ttfts[-1]*1000:.1f}ms, Prefill={input_len/ttfts[-1]:.0f} tok/s")
            else:
                print(f"  Run {i+1}: FAILED (no first token received)")

        if ttfts:
            ttfts_arr = np.array(ttfts)
            prefill_tps = input_len / ttfts_arr
            stats = {
                "input_len": actual_len,
                "ttft_ms_avg": float(np.mean(ttfts_arr) * 1000),
                "ttft_ms_std": float(np.std(ttfts_arr) * 1000),
                "ttft_ms_min": float(np.min(ttfts_arr) * 1000),
                "ttft_ms_max": float(np.max(ttfts_arr) * 1000),
                "prefill_tps_avg": float(np.mean(prefill_tps)),
                "prefill_tps_min": float(np.min(prefill_tps)),
                "prefill_tps_max": float(np.max(prefill_tps)),
                "all_ttfts_ms": [float(t * 1000) for t in ttfts],
            }
            all_results[name] = stats
            print(f"\nResults:")
            print(f"  TTFT: avg={stats['ttft_ms_avg']:.1f}ms, std={stats['ttft_ms_std']:.1f}ms")
            print(f"  Prefill: avg={stats['prefill_tps_avg']:.0f} tok/s, max={stats['prefill_tps_max']:.0f} tok/s")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Input':>8s} | {'TTFT avg(ms)':>12s} | {'TTFT std':>10s} | {'Prefill avg':>12s} | {'Prefill max':>12s}")
    print("-" * 80)
    for name, stats in all_results.items():
        print(f"{name:>8s} | {stats['ttft_ms_avg']:12.1f} | {stats['ttft_ms_std']:10.1f} | {stats['prefill_tps_avg']:12.0f} | {stats['prefill_tps_max']:12.0f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "engine": "Python TurboMind",
        "model": MODEL_PATH,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "warmup_runs": WARMUP,
            "measure_runs": NUM_REQUESTS,
            "max_prefill_token_num": 8192,
            "quant_policy": 4,
        },
        "results": all_results,
    }

    output_file = f"python_prefill_benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
