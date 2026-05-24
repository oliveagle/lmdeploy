#!/usr/bin/env python3
"""
gRPC Connection Pool Benchmark

Demonstrates the performance benefit of using a connection pool
for gRPC requests to the LMDeploy Rust server.

This benchmark compares:
1. No pool: Create a new channel for each request
2. With pool: Reuse a single channel across all requests

Expected results:
- Without pool: ~10-50ms connection overhead per request
- With pool: ~0ms overhead (connection reused after first request)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

import grpc

# Import generated protobuf modules
sys.path.insert(0, str(Path(__file__).parent))
try:
    from lmdeploy.v1 import lm_deploy_pb2, lm_deploy_pb2_grpc
except ImportError:
    print("Error: gRPC protobuf modules not available.")
    print("Make sure the protobuf files are generated:")
    print("  cd lmdeploy-rust-server && cargo build")
    sys.exit(1)

# Import connection pool
from grpc_pool import GrpcConnectionPool

# Test configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 50051
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
DEFAULT_MAX_TOKENS = 64
DEFAULT_NUM_REQUESTS = 10


def benchmark_without_pool(
    host: str,
    port: int,
    prompt: str,
    max_tokens: int,
    num_requests: int,
) -> Dict[str, Any]:
    """Benchmark gRPC requests without connection pooling.

    Each request creates a new gRPC channel, incurring connection overhead.
    """
    print(f"\n{'='*60}")
    print("Benchmark WITHOUT Connection Pool")
    print(f"{'='*60}")
    print(f"Server: {host}:{port}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Max tokens: {max_tokens}")
    print(f"Requests: {num_requests}")

    timings = []
    connection_overheads = []
    ttfts = []
    token_count = 0

    for i in range(num_requests):
        # Measure connection overhead
        conn_start = time.perf_counter()
        channel = grpc.insecure_channel(f"{host}:{port}")
        conn_time_ms = (time.perf_counter() - conn_start) * 1000
        connection_overheads.append(conn_time_ms)

        stub = lm_deploy_pb2_grpc.LmDeployServiceStub(channel)

        # Measure request latency (TTFT)
        request = lm_deploy_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )

        req_start = time.perf_counter()
        first_token_ms = None
        tokens = 0

        try:
            for response in stub.GenerateStream(request):
                if response.payload.chunk.text and response.payload.chunk.text != "[DONE]":
                    elapsed = (time.perf_counter() - req_start) * 1000
                    if first_token_ms is None:
                        first_token_ms = elapsed
                    tokens += 1

            total_ms = (time.perf_counter() - req_start) * 1000
            timings.append(total_ms)
            if first_token_ms is not None:
                ttfts.append(first_token_ms)
            token_count += tokens

            print(f"  Request {i+1:2d}: conn={conn_time_ms:6.2f}ms, "
                  f"ttft={first_token_ms or 0:6.2f}ms, total={total_ms:6.2f}ms")

        except grpc.RpcError as e:
            print(f"  Request {i+1:2d}: gRPC error: {e}")
            connection_overheads.pop()

        finally:
            channel.close()

    if not timings:
        return {"error": "All requests failed"}

    return {
        "mode": "without_pool",
        "num_requests": num_requests,
        "total_tokens": token_count,
        "avg_connection_overhead_ms": sum(connection_overheads) / len(connection_overheads),
        "min_connection_overhead_ms": min(connection_overheads),
        "max_connection_overhead_ms": max(connection_overheads),
        "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
        "avg_total_latency_ms": sum(timings) / len(timings),
        "total_time_ms": sum(timings),
    }


def benchmark_with_pool(
    host: str,
    port: int,
    prompt: str,
    max_tokens: int,
    num_requests: int,
) -> Dict[str, Any]:
    """Benchmark gRPC requests with connection pooling.

    All requests reuse the same gRPC channel, avoiding connection overhead.
    """
    print(f"\n{'='*60}")
    print("Benchmark WITH Connection Pool")
    print(f"{'='*60}")
    print(f"Server: {host}:{port}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Max tokens: {max_tokens}")
    print(f"Requests: {num_requests}")

    # Create pool and measure first connection
    pool_start = time.perf_counter()
    pool = GrpcConnectionPool(host=host, port=port)
    pool_creation_ms = (time.perf_counter() - pool_start) * 1000

    print(f"  Pool creation: {pool_creation_ms:.2f}ms")

    stub = pool.get_stub()
    timings = []
    ttfts = []
    token_count = 0

    for i in range(num_requests):
        request = lm_deploy_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )

        req_start = time.perf_counter()
        first_token_ms = None
        tokens = 0

        try:
            for response in stub.GenerateStream(request):
                if response.payload.chunk.text and response.payload.chunk.text != "[DONE]":
                    elapsed = (time.perf_counter() - req_start) * 1000
                    if first_token_ms is None:
                        first_token_ms = elapsed
                    tokens += 1

            total_ms = (time.perf_counter() - req_start) * 1000
            timings.append(total_ms)
            if first_token_ms is not None:
                ttfts.append(first_token_ms)
            token_count += tokens

            print(f"  Request {i+1:2d}: ttft={first_token_ms or 0:6.2f}ms, "
                  f"total={total_ms:6.2f}ms (no connection overhead)")

        except grpc.RpcError as e:
            print(f"  Request {i+1:2d}: gRPC error: {e}")

    pool.close()

    if not timings:
        return {"error": "All requests failed"}

    return {
        "mode": "with_pool",
        "num_requests": num_requests,
        "total_tokens": token_count,
        "pool_creation_ms": pool_creation_ms,
        "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
        "avg_total_latency_ms": sum(timings) / len(timings),
        "total_time_ms": sum(timings),
    }


def print_comparison(without_pool: Dict[str, Any], with_pool: Dict[str, Any]):
    """Print comparison of results."""
    print(f"\n{'='*80}")
    print("COMPARISON: Connection Pool Benefit")
    print(f"{'='*80}")

    if "error" in without_pool or "error" in with_pool:
        print("Error: Could not complete benchmark")
        return

    # Calculate overhead difference
    conn_overhead = without_pool["avg_connection_overhead_ms"]
    pool_overhead = with_pool["pool_creation_ms"] / with_pool["num_requests"]

    overhead_saved_ms = conn_overhead - pool_overhead
    overhead_saved_pct = (overhead_saved_ms / conn_overhead * 100) if conn_overhead > 0 else 0

    print(f"\nConnection Overhead:")
    print(f"  Without pool: {conn_overhead:.2f}ms per request")
    print(f"  With pool:    {pool_overhead:.2f}ms per request (amortized)")
    print(f"  Saved:        {overhead_saved_ms:+.2f}ms ({overhead_saved_pct:+.1f}%)")

    # TTFT comparison
    ttft_diff = with_pool["avg_ttft_ms"] - without_pool["avg_ttft_ms"]
    ttft_pct = (ttft_diff / without_pool["avg_ttft_ms"] * 100) if without_pool["avg_ttft_ms"] > 0 else 0

    print(f"\nTime To First Token (TTFT):")
    print(f"  Without pool: {without_pool['avg_ttft_ms']:.2f}ms")
    print(f"  With pool:    {with_pool['avg_ttft_ms']:.2f}ms")
    print(f"  Difference:   {ttft_diff:+.2f}ms ({ttft_pct:+.1f}%)")

    # Total throughput comparison
    total_without = without_pool["total_time_ms"]
    total_with = with_pool["total_time_ms"]
    total_diff = total_without - total_with
    total_pct = (total_diff / total_without * 100) if total_without > 0 else 0

    print(f"\nTotal Time ({with_pool['num_requests']} requests):")
    print(f"  Without pool: {total_without:.2f}ms")
    print(f"  With pool:    {total_with:.2f}ms")
    print(f"  Saved:        {total_diff:+.2f}ms ({total_pct:+.1f}%)")

    # Throughput improvement
    tps_without = with_pool["total_tokens"] / (total_without / 1000)
    tps_with = with_pool["total_tokens"] / (total_with / 1000)
    tps_improvement = ((tps_with - tps_without) / tps_without * 100) if tps_without > 0 else 0

    print(f"\nEffective Throughput:")
    print(f"  Without pool: {tps_without:.1f} tokens/sec")
    print(f"  With pool:    {tps_with:.1f} tokens/sec")
    print(f"  Improvement:  {tps_improvement:+.1f}%")

    print(f"\n{'='*80}")
    print("KEY FINDING:")
    if overhead_saved_ms > 5:
        print(f"✓ Connection pool saves ~{overhead_saved_ms:.1f}ms per request")
        print(f"✓ For high-frequency requests, this significantly improves throughput")
    else:
        print(f"✓ Connection overhead is low ({conn_overhead:.1f}ms)")
        print(f"✓ Pool still provides consistency and resource management benefits")
    print(f"{'='*80}")


def main():
    """Run the benchmark and print results."""
    parser = argparse.ArgumentParser(
        description="Benchmark gRPC connection pool performance"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"gRPC server host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"gRPC server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt text to use (default: {len(DEFAULT_PROMPT)} chars)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help=f"Number of requests to make (default: {DEFAULT_NUM_REQUESTS})",
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("gRPC Connection Pool Benchmark")
    print("=" * 80)

    # Run benchmarks
    without_pool = benchmark_without_pool(
        args.host,
        args.port,
        args.prompt,
        args.max_tokens,
        args.num_requests,
    )

    with_pool = benchmark_with_pool(
        args.host,
        args.port,
        args.prompt,
        args.max_tokens,
        args.num_requests,
    )

    # Print comparison
    print_comparison(without_pool, with_pool)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        results = {
            "config": {
                "host": args.host,
                "port": args.port,
                "prompt_length": len(args.prompt),
                "max_tokens": args.max_tokens,
                "num_requests": args.num_requests,
            },
            "without_pool": without_pool,
            "with_pool": with_pool,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
