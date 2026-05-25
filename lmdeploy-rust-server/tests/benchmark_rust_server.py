#!/usr/bin/env python3
"""
Rust Server Prefill Benchmark

Measures prefill performance of the Rust Server via gRPC.
Compares with Python TurboMind baseline.
"""

import argparse
import grpc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tests"))

try:
    from lmdeploy.v1 import lm_deploy_pb2, lm_deploy_pb2_grpc
except ImportError:
    print("Error: gRPC protobuf modules not available.")
    print("Run: cd lmdeploy-rust-server && cargo build")
    sys.exit(1)

# Configuration
MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "
WARMUP_RUNS = 2
MEASURE_RUNS = 3
OUTPUT_TOKENS = 1  # Only need first token for TTFT measurement


def gen_prompt(token_count: int) -> str:
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats


def run_rust_benchmark(host: str, port: int, ctx_tokens: int) -> Dict[str, Any]:
    """Benchmark Rust Server prefill via gRPC."""
    prompt = gen_prompt(ctx_tokens)

    # Connect to Rust server
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = lm_deploy_pb2_grpc.LmDeployServiceStub(channel)

    # Warmup
    print(f"  Warmup ({WARMUP_RUNS} runs)...", end=" ", flush=True)
    for _ in range(WARMUP_RUNS):
        req = lm_deploy_pb2.GenerateRequest(
            prompt=gen_prompt(1024),
            max_tokens=OUTPUT_TOKENS,
            temperature=0.7,
        )
        try:
            for _ in stub.GenerateStream(req):
                pass
        except grpc.RpcError:
            pass
    print("OK")

    # Measure
    ttfts = []
    for r in range(MEASURE_RUNS):
        req = lm_deploy_pb2.GenerateRequest(
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
            print(f"{first_token_ms:.0f}ms ", end="", flush=True)
        else:
            print("timeout ", end="", flush=True)

    channel.close()

    if ttfts:
        avg_ms = sum(ttfts) / len(ttfts)
        min_ms = min(ttfts)
        avg_tps = ctx_tokens / avg_ms * 1000 if avg_ms > 0 else 0
        max_tps = ctx_tokens / min_ms * 1000 if min_ms > 0 else 0
        return {
            "ctx_tokens": ctx_tokens,
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "avg_tps": avg_tps,
            "max_tps": max_tps,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Rust Server Prefill Benchmark")
    parser.add_argument("--host", default="localhost", help="Rust server host")
    parser.add_argument("--port", type=int, default=50051, help="Rust server port")
    args = parser.parse_args()

    print("=" * 80)
    print("Rust Server Prefill Benchmark")
    print("=" * 80)
    print(f"\nServer: {args.host}:{args.port}")
    print(f"Model: {MODEL}")
    print(f"Warmup: {WARMUP_RUNS}, Measure: {MEASURE_RUNS}")
    print(f"Output tokens: {OUTPUT_TOKENS}")

    test_contexts = [
        ("1K", 1024),
        ("2K", 2048),
        ("4K", 4096),
        ("8K", 8192),
    ]

    results = {}
    print("\n" + "=" * 80)
    print("Measuring prefill performance...")
    print("=" * 80)

    for label, ctx_tokens in test_contexts:
        print(f"\n{label} ({ctx_tokens} tokens): ", end="", flush=True)
        result = run_rust_benchmark(args.host, args.port, ctx_tokens)
        if result:
            results[label] = result
            print(f"| Avg: {result['avg_tps']:.0f} tok/s, Max: {result['max_tps']:.0f} tok/s")
        else:
            print("| FAILED")

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
    print("\n" + "=" * 80)
    print("Comparison with Python TurboMind Baseline")
    print("=" * 80)
    python_baseline = {
        "1K": {"avg_tps": 14827, "avg_ms": 69},
        "4K": {"avg_tps": 33727, "avg_ms": 122},
        "8K": {"avg_tps": 42875, "avg_ms": 191},
    }
    print(f"\n{'Context':>8} | {'Rust TPS':>12} | {'Python TPS':>12} | {'Ratio':>10}")
    print("-" * 60)
    for label in ["1K", "4K", "8K"]:
        if label in results and label in python_baseline:
            r = results[label]
            p = python_baseline[label]
            ratio = r['avg_tps'] / p['avg_tps'] if p['avg_tps'] > 0 else 0
            print(f"{label:>8} | {r['avg_tps']:>12.0f} | {p['avg_tps']:>12.0f} | {ratio:>10.1%}")

    # Save results
    output = {
        "engine": "Rust Server (PureCpp via gRPC)",
        "model": MODEL,
        "host": args.host,
        "port": args.port,
        "warmup_runs": WARMUP_RUNS,
        "measure_runs": MEASURE_RUNS,
        "results": results,
        "timestamp": time.time(),
    }
    out_path = Path(__file__).parent / "benchmark_rust_server.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
