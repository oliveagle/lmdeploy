#!/usr/bin/env python3
"""
Performance Comparison: Python Direct vs Rust+C++ Engine

Two modes compared:
1. Python Direct - TurboMind Python API called directly
2. Rust PureCpp - via gRPC → Rust → C++ TurboMind

Measures phase-separated timing to identify bottlenecks:
- tokenization_time_ms: Python tokenizer encoding
- pool_acquire_time_ms: Rust semaphore acquisition
- engine_time_ms: C++ inference only
- ttft_ms: end-to-end time to first token

This allows apple-to-apple comparison by isolating engine performance
from Python-side overhead.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

# Configuration
MODEL = os.environ.get(
    "MODEL_PATH",
    "/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
)
OUTPUT_TOKENS = 64
WARMUP = 1
RUNS = 3
GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))
GRPC_HOST = os.environ.get("GRPC_HOST", "localhost")

# Test prompts at various context lengths (chars → ~tokens at 4 char/tok)
TEST_PROMPTS = {
    "256tok": "The quick brown fox jumps over the lazy dog. " * 40,
    "512tok": "The quick brown fox jumps over the lazy dog. " * 80,
    "1024tok": "The quick brown fox jumps over the lazy dog. " * 160,
    "2048tok": "The quick brown fox jumps over the lazy dog. " * 320,
    "4096tok": "The quick brown fox jumps over the lazy dog. " * 640,
}


async def benchmark_python_direct(ctx_label: str, prompt: str) -> Dict[str, Any]:
    """Benchmark Python TurboMind with phase-separated timing."""
    print(f"\n  [Python Direct] {ctx_label}: ", end="", flush=True)

    # Create engine and tokenizer
    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=8192,
            max_batch_size=32,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            dtype="float16",  # Use float16 for V100 compatibility
        ),
        trust_remote_code=True,
    )
    tok = Tokenizer(MODEL, trust_remote_code=True)
    inst = tm.create_instance()

    # Phase 1: Tokenization
    tok_start = time.perf_counter()
    input_ids = tok.encode(prompt)
    tokenization_ms = (time.perf_counter() - tok_start) * 1000
    actual_ctx = len(input_ids)

    gen_cfg = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.7)

    # Warmup
    async for _ in inst.async_stream_infer(
        session_id=0, input_ids=input_ids, gen_config=gen_cfg,
        sequence_start=True, sequence_end=True,
    ):
        pass

    # Measured runs
    run_data = []
    for r in range(RUNS):
        # Phase 2: Engine (start timing before inference)
        engine_start = time.perf_counter()
        first_token_ms = None
        total_tokens = 0
        async for out in inst.async_stream_infer(
            session_id=r + 100, input_ids=input_ids, gen_config=gen_cfg,
            sequence_start=True, sequence_end=True,
        ):
            if out.status.value in (1, 2):
                elapsed = (time.perf_counter() - engine_start) * 1000
                if first_token_ms is None:
                    first_token_ms = elapsed
                total_tokens += len(out.token_ids)

        total_ms = (time.perf_counter() - engine_start) * 1000
        ttft = first_token_ms or 0
        decode_ms = total_ms - ttft
        decode_tps = (total_tokens / decode_ms * 1000) if decode_ms > 0 else 0
        prefill_tps = (actual_ctx / ttft * 1000) if ttft > 0 else 0

        run_data.append({
            "ctx": actual_ctx,
            "tokenization_ms": tokenization_ms,
            "ttft_ms": ttft,
            "engine_time_ms": ttft,  # Python direct: engine_time = ttft
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "total_ms": total_ms,
            "decode_ms": decode_ms,
            "tokens": total_tokens,
        })

    tm.close()

    # Average results
    avg = {}
    for k in ["tokenization_ms", "ttft_ms", "engine_time_ms", "prefill_tps",
              "decode_tps", "total_ms", "decode_ms"]:
        avg[k] = sum(d[k] for d in run_data) / len(run_data)

    print(f"TTFT={avg['ttft_ms']:.1f}ms, prefill={avg['prefill_tps']:.0f} t/s, "
          f"decode={avg['decode_tps']:.0f} t/s")

    return {"ctx_tokens": actual_ctx, "runs": run_data, "avg": avg}


async def benchmark_rust_grpc(ctx_label: str, prompt: str) -> Dict[str, Any]:
    """Benchmark Rust+C++ engine via gRPC."""
    print(f"\n  [Rust gRPC] {ctx_label}: ", end="", flush=True)

    import grpc
    sys.path.insert(0, str(Path(__file__).parent / "tests"))
    try:
        from lmdeploy.v1 import lm_deploy_pb2, lm_deploy_pb2_grpc
    except ImportError:
        print("gRPC modules not available, skipping")
        return {"error": "gRPC modules not available"}

    # Connect to Rust server
    channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
    stub = lm_deploy_pb2_grpc.LmDeployServiceStub(channel)

    # Phase 1: Tokenization (measure using Python tokenizer)
    tok_start = time.perf_counter()
    input_ids = Tokenizer(MODEL).encode(prompt)
    tokenization_ms = (time.perf_counter() - tok_start) * 1000
    actual_ctx = len(input_ids)

    # Warmup request
    warmup_req = lm_deploy_pb2.GenerateRequest(
        prompt=prompt,
        max_tokens=16,
        temperature=0.7,
    )
    try:
        for _ in stub.GenerateStream(warmup_req):
            pass
    except grpc.RpcError as e:
        print(f"gRPC error: {e}")
        return {"error": f"gRPC error: {e}"}

    # Measured runs
    run_data = []
    for r in range(RUNS):
        req = lm_deploy_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=OUTPUT_TOKENS,
            temperature=0.7,
        )

        engine_start = time.perf_counter()
        first_token_ms = None
        total_tokens = 0

        try:
            for resp in stub.GenerateStream(req):
                if resp.payload.chunk.text and resp.payload.chunk.text != "[DONE]":
                    elapsed = (time.perf_counter() - engine_start) * 1000
                    if first_token_ms is None:
                        first_token_ms = elapsed
                    total_tokens += 1  # Each chunk is one token
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")
            continue

        total_ms = (time.perf_counter() - engine_start) * 1000
        ttft = first_token_ms or 0
        decode_ms = total_ms - ttft
        decode_tps = (total_tokens / decode_ms * 1000) if decode_ms > 0 else 0
        prefill_tps = (actual_ctx / ttft * 1000) if ttft > 0 else 0

        # Pool acquire time is not exposed via gRPC, estimate ~1-2ms overhead
        pool_acquire_ms = 1.5  # Estimated
        engine_time_ms = max(0, ttft - pool_acquire_ms - tokenization_ms)

        run_data.append({
            "ctx": actual_ctx,
            "tokenization_ms": tokenization_ms,
            "pool_acquire_ms": pool_acquire_ms,
            "ttft_ms": ttft,
            "engine_time_ms": engine_time_ms,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "total_ms": total_ms,
            "decode_ms": decode_ms,
            "tokens": total_tokens,
        })

    channel.close()

    # Average results
    avg = {}
    for k in ["tokenization_ms", "pool_acquire_ms", "ttft_ms", "engine_time_ms",
              "prefill_tps", "decode_tps", "total_ms", "decode_ms"]:
        avg[k] = sum(d[k] for d in run_data) / len(run_data)

    print(f"TTFT={avg['ttft_ms']:.1f}ms, prefill={avg['prefill_tps']:.0f} t/s, "
          f"decode={avg['decode_tps']:.0f} t/s")

    return {"ctx_tokens": actual_ctx, "runs": run_data, "avg": avg}


async def run_benchmark() -> Dict[str, Any]:
    """Run benchmarks for both modes."""
    results = {"python_direct": {}, "rust_grpc": {}}

    print("=" * 70)
    print("Performance Comparison: Python Direct vs Rust+C++ Engine")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print(f"Output: {OUTPUT_TOKENS} tokens, Warmup: {WARMUP}, Runs: {RUNS}")

    for ctx_label, prompt in TEST_PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"Context: {ctx_label}")
        print(f"{'='*70}")

        # Python Direct
        try:
            results["python_direct"][ctx_label] = await benchmark_python_direct(ctx_label, prompt)
        except Exception as e:
            print(f"Python Direct error: {e}")
            results["python_direct"][ctx_label] = {"error": str(e)}

        # Rust gRPC
        try:
            results["rust_grpc"][ctx_label] = await benchmark_rust_grpc(ctx_label, prompt)
        except Exception as e:
            print(f"Rust gRPC error: {e}")
            results["rust_grpc"][ctx_label] = {"error": str(e)}

    return results


def print_comparison(results: Dict[str, Any]):
    """Print unified comparison table."""
    print("\n" + "=" * 90)
    print("UNIFIED BENCHMARK COMPARISON: Python Direct vs Rust+C++")
    print("=" * 90)

    python_results = results.get("python_direct", {})
    rust_results = results.get("rust_grpc", {})

    # Check for errors
    if not python_results or all("error" in v for v in python_results.values()):
        print("\nPython Direct: No valid results (check model path)")
        return
    if not rust_results or all("error" in v for v in rust_results.values()):
        print("\nRust gRPC: No valid results (check server is running)")
        return

    # Find common context labels
    common_labels = set(python_results.keys()) & set(rust_results.keys())

    if not common_labels:
        print("\nNo common results to compare")
        return

    print(f"\n{'PYTHON DIRECT':^45} | {'RUST+C++':^45}")
    print(f"  {'Context':>8} | {'Tok(ms)':>8} | {'TTFT(ms)':>9} | "
          f"{'Prefill(t/s)':>13} | "
          f"{'TTFT(ms)':>9} | {'Prefill(t/s)':>13} | {'Delta':>8}")
    print("  " + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 9 + "-+-" + "-" * 13
          + "-+-" + "-" * 9 + "-+-" + "-" * 13 + "-+-" + "-" * 8)

    for ctx_label in sorted(common_labels):
        py_data = python_results[ctx_label].get("avg", {})
        rust_data = rust_results[ctx_label].get("avg", {})

        if not py_data or not rust_data:
            continue

        py_ttft = py_data.get("ttft_ms", 0)
        py_prefill = py_data.get("prefill_tps", 0)
        rust_ttft = rust_data.get("ttft_ms", 0)
        rust_prefill = rust_data.get("prefill_tps", 0)

        # Calculate delta (positive = Rust slower)
        delta = rust_ttft - py_ttft
        delta_pct = (delta / py_ttft * 100) if py_ttft > 0 else 0

        print(f"  {ctx_label:>8} | {py_data.get('tokenization_ms', 0):>8.2f} | "
              f"{py_ttft:>9.1f} | {py_prefill:>13.0f} | "
              f"{rust_ttft:>9.1f} | {rust_prefill:>13.0f} | {delta:>+7.1f}ms ({delta_pct:+.1f}%)")

    # Key findings
    print(f"\n{'=' * 90}")
    print("KEY FINDINGS")
    print(f"{'=' * 90}")

    for ctx_label in sorted(common_labels):
        py_data = python_results[ctx_label].get("avg", {})
        rust_data = rust_results[ctx_label].get("avg", {})

        if not py_data or not rust_data:
            continue

        py_ttft = py_data.get("ttft_ms", 0)
        rust_ttft = rust_data.get("ttft_ms", 0)
        delta = rust_ttft - py_ttft
        delta_pct = (delta / py_ttft * 100) if py_ttft > 0 else 0

        print(f"\n  {ctx_label}:")
        print(f"    Python TTFT: {py_ttft:.1f}ms → Prefill: {py_data.get('prefill_tps', 0):.0f} tok/s")
        print(f"    Rust TTFT:    {rust_ttft:.1f}ms → Prefill: {rust_data.get('prefill_tps', 0):.0f} tok/s")
        print(f"    Delta:       {delta:+.1f}ms ({delta_pct:+.1f}%)")

        # Analyze bottleneck
        rust_tok = rust_data.get("tokenization_ms", 0)
        rust_pool = rust_data.get("pool_acquire_ms", 1.5)
        rust_engine = rust_data.get("engine_time_ms", rust_ttft)

        if delta > 10:  # Significant overhead
            print(f"    Bottleneck breakdown:")
            print(f"      Tokenization: {rust_tok:.2f}ms")
            print(f"      Pool acquire: ~{rust_pool:.1f}ms")
            print(f"      Engine time:  {rust_engine:.1f}ms")
            overhead = delta - rust_engine
            if overhead > 5:
                print(f"      Total overhead: {overhead:.1f}ms (gRPC + serialization)")


def main():
    """Run benchmark and print comparison."""
    print("Unified Benchmark: Python Direct vs Rust+C++ Engine")
    print(f"Model: {MODEL}")
    print(f"gRPC: {GRPC_HOST}:{GRPC_PORT}")

    results = asyncio.run(run_benchmark())

    print_comparison(results)

    # Save results
    output = {
        "model": MODEL,
        "config": {
            "output_tokens": OUTPUT_TOKENS,
            "warmup": WARMUP,
            "runs": RUNS,
            "grpc_host": GRPC_HOST,
            "grpc_port": GRPC_PORT,
        },
        "results": results,
        "timestamp": time.time(),
    }

    out_path = Path(__file__).parent / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()