#!/usr/bin/env python3
"""
统一 Prefill 性能对比测试 - Python TurboMind vs Rust Server
测量相同模型和输入下的 TTFT 和 prefill throughput
"""
import asyncio
import json
import time
from datetime import datetime
import numpy as np
import subprocess
import os

MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
RUST_SERVER_URL = "http://localhost:8000"  # 假设 Rust server 运行在此端口

# Test scenarios
SCENARIOS = [
    {"name": "512", "input_len": 512},
    {"name": "1024", "input_len": 1024},
    {"name": "2048", "input_len": 2048},
    {"name": "4096", "input_len": 4096},
    {"name": "8192", "input_len": 8192},
]

NUM_REQUESTS = 5
WARMUP_REQUESTS = 2


def generate_prompt_ids(target_len):
    """Generate repeated token IDs for a target length"""
    base_ids = [12800, 12801, 12802, 12803, 12804, 12805, 12806, 12807, 12808, 12809]
    repeat = (target_len // len(base_ids)) + 1
    return (base_ids * repeat)[:target_len]


async def benchmark_python():
    """Benchmark Python TurboMind prefill performance"""
    print("\n" + "="*60)
    print("Python TurboMind Prefill Benchmark")
    print(f"Model: {MODEL_PATH}")
    print("="*60)

    try:
        from lmdeploy.turbomind import TurboMind
        from lmdeploy.messages import TurbomindEngineConfig, GenerationConfig
    except ImportError:
        print("ERROR: lmdeploy not installed, skipping Python benchmark")
        return None

    # Initialize engine with same config as Rust server
    engine_config = TurbomindEngineConfig(
        session_len=154880,
        max_batch_size=32,
        cache_max_entry_count=0.5,
        max_prefill_token_num=8192,
        quant_policy=4,  # AWQ
        enable_prefix_caching=False,
    )

    print("Loading model...")
    tm_model = TurboMind(MODEL_PATH, engine_config=engine_config)
    model_inst = tm_model.create_instance()

    results = {}

    for scenario in SCENARIOS:
        name = scenario["name"]
        input_len = scenario["input_len"]
        input_ids = generate_prompt_ids(input_len)
        actual_len = len(input_ids)

        print(f"\n--- Scenario: {name} ({actual_len} tokens) ---")

        # Warmup
        for _ in range(WARMUP_REQUESTS):
            gen = model_inst.async_stream_infer(
                0,
                input_ids=[1, 2, 3],
                gen_config=GenerationConfig(max_new_tokens=1, ignore_eos=True),
                stream_output=True,
            )
            async for _ in gen:
                pass
            await gen.aclose()

        # Benchmark
        ttfts = []
        for i in range(NUM_REQUESTS):
            start = time.perf_counter()
            first_token = True
            token_count = 0

            gen = model_inst.async_stream_infer(
                i + 100,
                input_ids=input_ids,
                gen_config=GenerationConfig(max_new_tokens=1, ignore_eos=True),
                stream_output=True,
            )
            async for outputs in gen:
                if first_token:
                    ttft = time.perf_counter() - start
                    ttfts.append(ttft)
                    first_token = False
                token_count += 1
            await gen.aclose()

        if ttfts:
            ttfts_arr = np.array(ttfts)
            prefill_tps = actual_len / ttfts_arr
            stats = {
                "input_len": actual_len,
                "ttft_ms_avg": float(np.mean(ttfts_arr) * 1000),
                "ttft_ms_std": float(np.std(ttfts_arr) * 1000),
                "ttft_ms_min": float(np.min(ttfts_arr) * 1000),
                "ttft_ms_max": float(np.max(ttfts_arr) * 1000),
                "prefill_tps_avg": float(np.mean(prefill_tps)),
                "prefill_tps_min": float(np.min(prefill_tps)),
                "prefill_tps_max": float(np.max(prefill_tps)),
            }
            results[name] = stats
            print(f"  TTFT: avg={stats['ttft_ms_avg']:.1f}ms, std={stats['ttft_ms_std']:.1f}ms")
            print(f"  Prefill: avg={stats['prefill_tps_avg']:.0f} tok/s, max={stats['prefill_tps_max']:.0f} tok/s")

    return results


async def benchmark_rust():
    """Benchmark Rust server prefill performance via HTTP API"""
    print("\n" + "="*60)
    print("Rust Server Prefill Benchmark")
    print(f"Server: {RUST_SERVER_URL}")
    print("="*60)

    try:
        import httpx
    except ImportError:
        print("ERROR: httpx not installed, skipping Rust benchmark")
        return None

    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{RUST_SERVER_URL}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"WARNING: Server health check failed (status={resp.status_code})")
    except Exception as e:
        print(f"WARNING: Cannot connect to Rust server at {RUST_SERVER_URL}: {e}")
        print("Skipping Rust benchmark. Start the server first.")
        return None

    results = {}

    for scenario in SCENARIOS:
        name = scenario["name"]
        input_len = scenario["input_len"]
        input_ids = generate_prompt_ids(input_len)
        actual_len = len(input_ids)

        print(f"\n--- Scenario: {name} ({actual_len} tokens) ---")

        # Warmup
        for _ in range(WARMUP_REQUESTS):
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{RUST_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                    timeout=60.0,
                )

        # Benchmark
        ttfts = []
        for _ in range(NUM_REQUESTS):
            start = time.perf_counter()
            first_token = True

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{RUST_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [{"role": "user", "content": " ".join(["token"] * actual_len)}],
                        "max_tokens": 1,
                        "stream": True,
                    },
                    timeout=300.0,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        if first_token:
                            ttft = time.perf_counter() - start
                            ttfts.append(ttft)
                            first_token = False

        if ttfts:
            ttfts_arr = np.array(ttfts)
            prefill_tps = actual_len / ttfts_arr
            stats = {
                "input_len": actual_len,
                "ttft_ms_avg": float(np.mean(ttfts_arr) * 1000),
                "ttft_ms_std": float(np.std(ttfts_arr) * 1000),
                "ttft_ms_min": float(np.min(ttfts_arr) * 1000),
                "ttft_ms_max": float(np.max(ttfts_arr) * 1000),
                "prefill_tps_avg": float(np.mean(prefill_tps)),
                "prefill_tps_min": float(np.min(prefill_tps)),
                "prefill_tps_max": float(np.max(prefill_tps)),
            }
            results[name] = stats
            print(f"  TTFT: avg={stats['ttft_ms_avg']:.1f}ms, std={stats['ttft_ms_std']:.1f}ms")
            print(f"  Prefill: avg={stats['prefill_tps_avg']:.0f} tok/s, max={stats['prefill_tps_max']:.0f} tok/s")

    return results


def print_comparison(python_results, rust_results):
    """Print side-by-side comparison"""
    if not python_results and not rust_results:
        print("No results to compare")
        return

    print("\n" + "="*100)
    print("PREFILL PERFORMANCE COMPARISON")
    print("="*100)

    header = f"{'Input':>8s} | "
    if python_results:
        header += f"{'Py TTFT(ms)':>12s} | {'Py Prefill':>12s} | "
    if rust_results:
        header += f"{'Rs TTFT(ms)':>12s} | {'Rs Prefill':>12s} | {'Ratio':>8s}"
    print(header)
    print("-" * len(header))

    for name in SCENARIOS:
        n = name["name"]
        row = f"{n:>8s} | "
        if python_results and n in python_results:
            p = python_results[n]
            row += f"{p['ttft_ms_avg']:12.1f} | {p['prefill_tps_avg']:12.0f} | "
        elif python_results:
            row += f"{'N/A':>12s} | {'N/A':>12s} | "

        if rust_results and n in rust_results:
            r = rust_results[n]
            ratio = ""
            if python_results and n in python_results:
                ratio_val = r['prefill_tps_avg'] / python_results[n]['prefill_tps_avg']
                ratio = f"{ratio_val:.2f}x"
            row += f"{r['ttft_ms_avg']:12.1f} | {r['prefill_tps_avg']:12.0f} | {ratio:>8s}"
        elif rust_results:
            row += f"{'N/A':>12s} | {'N/A':>12s} | {'N/A':>8s}"

        print(row)


async def main():
    print(f"Prefill Performance Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_PATH}")

    python_results = await benchmark_python()
    rust_results = await benchmark_rust()

    print_comparison(python_results, rust_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "model": MODEL_PATH,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "warmup_runs": WARMUP_REQUESTS,
            "measure_runs": NUM_REQUESTS,
            "max_prefill_token_num": 8192,
            "quant_policy": 4,
        },
        "python": python_results,
        "rust": rust_results,
    }

    output_file = f"benchmark_prefill_comparison_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
