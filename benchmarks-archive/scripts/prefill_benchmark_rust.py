#!/usr/bin/env python3
"""
Rust Server Prefill Benchmark

Measures prefill performance of the Rust Server via gRPC.
Matches the Python benchmark format for direct comparison.

Usage:
    python3 tests/prefill_benchmark_rust.py

Results are saved to: tests/prefill_benchmark_rust.json
"""

import argparse
import grpc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add the project path for protobuf imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lmdeploy_pb2 import GenerateRequest
    from lmdeploy_pb2_grpc import LMDeployServiceStub
except ImportError:
    print("Error: gRPC protobuf modules not available.")
    print("Run: python3 -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/lmdeploy.proto")
    sys.exit(1)

# Configuration - matches Python benchmark
MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "
WARMUP_RUNS = 2
MEASURE_RUNS = 5
OUTPUT_TOKENS = 1  # Only need first token for TTFT measurement

# Test contexts - matches Python benchmark
TEST_CONTEXTS = {
    "1K": 1000,
    "2K": 2000,
    "4K": 4000,
    "8K": 8000,
}


def gen_prompt(token_count: int) -> str:
    """Generate a prompt with approximately the specified token count."""
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats


def benchmark_context(
    stub,
    ctx_tokens: int,
    warmup_runs: int = WARMUP_RUNS,
    measure_runs: int = MEASURE_RUNS,
) -> Dict[str, Any]:
    """Benchmark a single context length."""
    prompt = gen_prompt(ctx_tokens)

    # Warmup
    for _ in range(warmup_runs):
        warmup_prompt = gen_prompt(1024)
        req = GenerateRequest(
            prompt=warmup_prompt,
            max_tokens=OUTPUT_TOKENS,
            temperature=0.7,
        )
        try:
            for _ in stub.GenerateStream(req):
                pass
        except grpc.RpcError:
            pass

    # Measure TTFT (Time To First Token)
    ttfts = []
    for r in range(measure_runs):
        req = GenerateRequest(
            prompt=prompt,
            max_tokens=OUTPUT_TOKENS,
            temperature=0.7,
        )

        start = time.perf_counter()
        first_token_ms = None

        try:
            for resp in stub.GenerateStream(req):
                if resp.payload.chunk.text and resp.payload.chunk.text != "[DONE]":
                    first_token_ms = (time.perf_counter() - start) * 1000
                    break
        except grpc.RpcError as e:
            print(f"\n  ERROR: {e}", end=" ")
            return None

        if first_token_ms:
            ttfts.append(first_token_ms)

    if not ttfts:
        return None

    avg_ms = sum(ttfts) / len(ttfts)
    min_ms = min(ttfts)
    max_ms = max(ttfts)
    avg_tps = ctx_tokens / avg_ms * 1000 if avg_ms > 0 else 0
    max_tps = ctx_tokens / min_ms * 1000 if min_ms > 0 else 0

    return {
        "ctx_tokens": ctx_tokens,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "avg_tps": avg_tps,
        "max_tps": max_tps,
    }


def run_benchmark(host: str, port: int) -> Dict[str, Any]:
    """Run the full benchmark suite against a Rust server."""
    # Connect to Rust server
    channel = grpc.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ],
    )
    stub = LMDeployServiceStub(channel)

    results = {}

    for label, ctx_tokens in TEST_CONTEXTS.items():
        print(f"\n  {label} ({ctx_tokens} tokens)... ", end="", flush=True)
        result = benchmark_context(stub, ctx_tokens)

        if result:
            results[label] = result
            print(f"Avg {result['avg_tps']:.0f} tok/s, Max {result['max_tps']:.0f} tok/s (TTFT: {result['avg_ms']:.1f}ms)")
        else:
            print("FAILED")
            return None

    channel.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rust Server Prefill Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to default localhost:50051
  python3 tests/prefill_benchmark_rust.py

  # Connect to custom host and port
  python3 tests/prefill_benchmark_rust.py --host 192.168.1.100 --port 23333

  # Custom number of runs
  python3 tests/prefill_benchmark_rust.py --warmup 3 --measure 10
        """,
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Rust server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Rust server port (default: 50051)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
        help=f"Number of warmup runs (default: {WARMUP_RUNS})",
    )
    parser.add_argument(
        "--measure",
        type=int,
        default=MEASURE_RUNS,
        help=f"Number of measurement runs (default: {MEASURE_RUNS})",
    )
    parser.add_argument(
        "--output",
        default="tests/prefill_benchmark_rust.json",
        help="Output JSON file path (default: tests/prefill_benchmark_rust.json)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with Python benchmark results",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Rust Server Prefill Benchmark")
    print("=" * 80)
    print(f"\nServer: {args.host}:{args.port}")
    print(f"Model: {MODEL}")
    print(f"Warmup: {args.warmup}, Measure: {args.measure}")
    print(f"Output tokens: {OUTPUT_TOKENS}")

    print("\n" + "=" * 80)
    print("Running benchmark...")
    print("=" * 80)

    results = run_benchmark(args.host, args.port)

    if results is None:
        print("\nBenchmark failed!")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 80)
    print("Summary - Rust Server Prefill Performance")
    print("=" * 80)
    print(f"\n{'Context':>8} | {'Avg TTFT':>10} | {'Avg TPS':>12} | {'Max TPS':>12}")
    print("-" * 60)
    for label in ["1K", "2K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"{label:>8} | {r['avg_ms']:>10.1f}ms | {r['avg_tps']:>12.0f} | {r['max_tps']:>12.0f}")

    # Comparison with Python baseline
    python_path = Path("tests/prefill_benchmark_real.json")
    if args.compare and python_path.exists():
        with open(python_path) as f:
            python_data = json.load(f)

        print("\n" + "=" * 80)
        print("Comparison with Python TurboMind Baseline")
        print("=" * 80)
        print(f"\n{'Context':>8} | {'Rust TPS':>12} | {'Python TPS':>12} | {'Ratio':>10}")
        print("-" * 60)
        for label in ["1K", "2K", "4K", "8K"]:
            if label in results and label in python_data.get("results", {}):
                r = results[label]
                p = python_data["results"][label]
                ratio = r['avg_tps'] / p['avg_tps'] if p['avg_tps'] > 0 else 0
                diff = r['avg_tps'] - p['avg_tps']
                diff_str = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
                print(f"{label:>8} | {r['avg_tps']:>12.0f} | {p['avg_tps']:>12.0f} | {ratio:>10.1%} ({diff_str})")

    # Save results
    output = {
        "engine": "Rust Server (PureCpp via gRPC)",
        "model": MODEL,
        "host": args.host,
        "port": args.port,
        "warmup_runs": args.warmup,
        "measure_runs": args.measure,
        "results": results,
        "timestamp": time.time(),
    }

    out_path = Path(__file__).parent / args.output
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()