#!/usr/bin/env python3
"""
Unified Benchmark Script for LMDeploy Python vs Rust
Matches the exact parameters and methodology used in the Python baseline tests.

Usage:
    python unified_benchmark.py --model /path/to/model --backend turbomind

Note: This script uses the REST API server approach (same as benchmark_turbomind_quick.py)
to ensure consistent testing across Python and Rust implementations.
"""

import argparse
import json
import time
import sys
import asyncio
import subprocess
import requests
import aiohttp
from pathlib import Path
from datetime import datetime

# Unified test configuration (matches Python baseline exactly)
UNIFIED_INPUT_LENGTHS = [512, 1024, 4096, 8192]
UNIFIED_OUTPUT_LENGTH = 512
UNIFIED_WARMUP_RUNS = 2
UNIFIED_MEASURE_RUNS = 5
UNIFIED_CONCURRENCY = 1  # Serial testing
DEFAULT_PORT = 23333

# Model path (can be overridden)
DEFAULT_MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"


def start_server(model_path: str, port: int, backend: str = "turbomind"):
    """Start TurboMind API server."""
    cmd = [
        "lmdeploy", "serve", "api_server",
        model_path,
        "--backend", backend,
        "--server-name", "0.0.0.0",
        "--server-port", str(port),
    ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_server_ready(port: int, max_wait: int = 120) -> bool:
    """Wait for server to be ready."""
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    print(f"Server ready. Model: {models[0]['id']}")
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_server(proc):
    """Stop the server."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


async def run_single_request(session, port: int, prompt: str, output_len: int):
    """Execute a single request and measure TTFT and decode metrics."""
    url = f"http://localhost:{port}/v1/chat/completions"

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
    }

    ttft = None
    itls = []
    total_time = None
    generated_text = ""
    last_ts = None

    try:
        start = time.perf_counter()

        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"  Error: status={resp.status}, body={text[:200]}")
                return None

            async for chunk_bytes in resp.content:
                chunk = chunk_bytes.decode().strip()
                if not chunk or chunk == "data: " or chunk == "data:":
                    continue

                if chunk.startswith("data: "):
                    chunk = chunk[6:]

                if chunk == "[DONE]":
                    total_time = time.perf_counter() - start
                    break

                try:
                    data = json.loads(chunk)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                        else:
                            if last_ts is not None:
                                itls.append(now - last_ts)
                        last_ts = now
                        generated_text += content

                except json.JSONDecodeError:
                    continue

            if total_time is None:
                total_time = time.perf_counter() - start

    except Exception as e:
        print(f"Request error: {e}")
        return None

    return {
        "ttft": ttft,
        "itls": itls,
        "total_time": total_time,
        "generated_len": len(generated_text),
    }


async def run_benchmark(port: int, scenarios: list, warmup_runs: int, measure_runs: int):
    """Run benchmarks for all scenarios."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        trust_remote_code=True
    )

    results = []

    async with aiohttp.ClientSession() as session:
        for scenario in scenarios:
            input_len = scenario["input_len"]
            output_len = scenario["output_len"]

            # Generate prompt with exact token count
            base_tokens = tokenizer.encode("Hello, how are you today?")
            repeat_times = (input_len // len(base_tokens)) + 1
            input_ids = (base_tokens * repeat_times)[:input_len]
            prompt = tokenizer.decode(input_ids)
            actual_tokens = len(tokenizer.encode(prompt))

            print(f"\nTesting: input_len={actual_tokens}, output_len={output_len}")

            # Warmup
            print(f"  Warmup ({warmup_runs} runs)...", end=" ", flush=True)
            for _ in range(warmup_runs):
                await run_single_request(session, port, prompt[:500], 32)
            print("OK")
            await asyncio.sleep(2)

            # Measure
            print(f"  Measuring ({measure_runs} runs)...", end=" ", flush=True)
            tasks = []
            for _ in range(measure_runs):
                tasks.append(run_single_request(session, port, prompt, output_len))

            all_results = await asyncio.gather(*tasks)
            print("OK")

            # Calculate statistics
            valid_results = [r for r in all_results if r is not None]
            if not valid_results:
                print("  No valid results!")
                continue

            ttfts = [r["ttft"] for r in valid_results if r["ttft"] is not None]
            itls_flat = [itl for r in valid_results for itl in r["itls"]]
            total_times = [r["total_time"] for r in valid_results if r["total_time"] is not None]

            # Decode time = total_time - ttft
            decode_times = []
            for r in valid_results:
                if r["ttft"] is not None and r["total_time"] is not None:
                    decode_time = r["total_time"] - r["ttft"]
                    decode_times.append(decode_time)

            # Calculate throughput (matching Python baseline formula)
            total_prefill_time = sum(ttfts) if ttfts else 0
            prefill_tps = len(valid_results) * actual_tokens / total_prefill_time if total_prefill_time > 0 else 0

            total_decode_time = sum(decode_times) if decode_times else 0
            decode_tps = len(valid_results) * output_len / total_decode_time if total_decode_time > 0 else 0

            result = {
                "input_len": actual_tokens,
                "output_len": output_len,
                "ttft_ms": sum(ttfts) / len(ttfts) * 1000 if ttfts else 0,
                "ttft_ms_p99": sorted(ttfts)[int(len(ttfts) * 0.99)] * 1000 if ttfts else 0,
                "decode_time_ms": sum(decode_times) / len(decode_times) * 1000 if decode_times else 0,
                "total_time_ms": sum(total_times) / len(total_times) * 1000 if total_times else 0,
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
            }
            results.append(result)

            print(f"  Results: TTFT={result['ttft_ms']:.1f}ms, "
                  f"Prefill={prefill_tps:.0f} tok/s, Decode={decode_tps:.1f} tok/s")

    return results


def run_python_benchmark(model_path: str, output_dir: str, port: int = DEFAULT_PORT):
    """Run Python TurboMind benchmark with unified parameters."""
    print("=" * 80)
    print("LMDeploy Python TurboMind Unified Benchmark")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Input lengths: {UNIFIED_INPUT_LENGTHS}")
    print(f"Output length: {UNIFIED_OUTPUT_LENGTH}")
    print(f"Warmup runs: {UNIFIED_WARMUP_RUNS}")
    print(f"Measure runs: {UNIFIED_MEASURE_RUNS}")
    print(f"Concurrency: {UNIFIED_CONCURRENCY}")
    print("=" * 80)
    print()

    # Start server
    print("[1/4] Starting TurboMind server...")
    proc = start_server(model_path, port)
    print()

    try:
        # Wait for server to be ready
        print("[2/4] Waiting for server to be ready...")
        if not wait_server_ready(port):
            print("Server failed to start!")
            return None
        print()

        # Define scenarios
        scenarios = [
            {"input_len": input_len, "output_len": UNIFIED_OUTPUT_LENGTH}
            for input_len in UNIFIED_INPUT_LENGTHS
        ]

        # Run benchmark
        print("[3/4] Running benchmarks...")
        results = asyncio.run(run_benchmark(port, scenarios, UNIFIED_WARMUP_RUNS, UNIFIED_MEASURE_RUNS))
        print()

        # Save results
        print("[4/4] Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"unified_benchmark_python_{timestamp}.json"

        output = {
            "engine": "Python TurboMind",
            "model": model_path,
            "backend": "turbomind",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "input_lengths": UNIFIED_INPUT_LENGTHS,
                "output_length": UNIFIED_OUTPUT_LENGTH,
                "warmup_runs": UNIFIED_WARMUP_RUNS,
                "measure_runs": UNIFIED_MEASURE_RUNS,
                "concurrency": UNIFIED_CONCURRENCY,
            },
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {output_file}")
        print()

        # Print summary table
        print("=" * 100)
        print("SUMMARY - Python TurboMind Unified Benchmark")
        print("=" * 100)
        print(f"{'Input':>8} | {'Output':>8} | {'TTFT (ms)':>12} | {'Prefill (tok/s)':>18} | {'Decode (tok/s)':>18}")
        print("-" * 100)
        for r in results:
            print(f"{r['input_len']:>8} | {r['output_len']:>8} | {r['ttft_ms']:>12.1f} | {r['prefill_tps']:>18.0f} | {r['decode_tps']:>18.1f}")
        print("=" * 100)

        return output_file

    finally:
        stop_server(proc)


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark for LMDeploy Python vs Rust"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks-archive/unified_python_rust_20260528/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="turbomind",
        choices=["turbomind", "pytorch"],
        help="Backend to use"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for API server"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run benchmark
    result_file = run_python_benchmark(args.model, args.output_dir, args.port)

    if result_file:
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {result_file}")
        return 0
    else:
        print("\nBenchmark failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
