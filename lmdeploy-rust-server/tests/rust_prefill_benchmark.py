#!/usr/bin/env python3
"""
Rust Server Prefill Benchmark
直接测试 Rust server 的 prefill 性能
"""
import asyncio
import json
import time
from datetime import datetime
import numpy as np

SERVER_URL = "http://localhost:8000"
MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

SCENARIOS = [
    {"name": "512", "input_len": 512},
    {"name": "1K", "input_len": 1024},
    {"name": "2K", "input_len": 2048},
    {"name": "4K", "input_len": 4096},
    {"name": "8K", "input_len": 8192},
]

NUM_REQUESTS = 5
WARMUP = 2


async def benchmark_rust():
    """Benchmark Rust server via HTTP API"""
    print("=" * 60)
    print("Rust Server Prefill Benchmark")
    print(f"Server: {SERVER_URL}")
    print("=" * 60)

    try:
        import httpx
    except ImportError:
        print("ERROR: httpx not installed")
        return None

    # Check server
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{SERVER_URL}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"Server health check failed: {resp.status_code}")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return None

    results = {}

    for scenario in SCENARIOS:
        name = scenario["name"]
        input_len = scenario["input_len"]

        print(f"\n--- {name} ({input_len} tokens) ---")

        # Generate prompt (repeated words for token count)
        words = ["hello", "world", "test", "data"]
        word_count = input_len // len("hello world test data ")
        prompt = " ".join(words) * (word_count // 4 + 1)
        prompt = prompt[:input_len * 4]  # Approximate

        # Warmup
        for _ in range(WARMUP):
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    json={"model": "default", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=60.0,
                )

        # Benchmark
        ttfts = []
        for i in range(NUM_REQUESTS):
            start = time.perf_counter()
            first_token = True

            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream(
                        "POST",
                        f"{SERVER_URL}/v1/chat/completions",
                        json={
                            "model": "default",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 1,
                            "stream": True,
                        },
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            if first_token:
                                ttft = time.perf_counter() - start
                                ttfts.append(ttft)
                                first_token = False
                                break
            except Exception as e:
                print(f"  Request {i+1} failed: {e}")
                continue

        if ttfts:
            ttfts_arr = np.array(ttfts)
            prefill_tps = input_len / ttfts_arr
            stats = {
                "input_len": input_len,
                "ttft_ms_avg": float(np.mean(ttfts_arr) * 1000),
                "ttft_ms_std": float(np.std(ttfts_arr) * 1000),
                "ttft_ms_min": float(np.min(ttfts_arr) * 1000),
                "ttft_ms_max": float(np.max(ttfts_arr) * 1000),
                "prefill_tps_avg": float(np.mean(prefill_tps)),
                "prefill_tps_max": float(np.max(prefill_tps)),
            }
            results[name] = stats
            print(f"  TTFT: avg={stats['ttft_ms_avg']:.1f}ms, std={stats['ttft_ms_std']:.1f}ms")
            print(f"  Prefill: avg={stats['prefill_tps_avg']:.0f} tok/s, max={stats['prefill_tps_max']:.0f} tok/s")

    return results


async def main():
    print(f"Rust Server Prefill Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = await benchmark_rust()

    if results:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Context':>8s} | {'TTFT(avg)':>12s} | {'Prefill':>12s}")
        print("-" * 40)
        for name, stats in results.items():
            print(f"{name:>8s} | {stats['ttft_ms_avg']:12.1f} | {stats['prefill_tps_avg']:12.0f}")

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "server": SERVER_URL,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        with open(f"rust_prefill_benchmark_{timestamp}.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to rust_prefill_benchmark_{timestamp}.json")


if __name__ == "__main__":
    asyncio.run(main())