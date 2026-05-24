#!/usr/bin/env python3
"""
Unified benchmark: Python TurboMind vs Rust C++ engine with detailed timing breakdown.
"""
import sys
sys.path.insert(0, '/mnt/data/lmdeploy')

import time
import json
import subprocess
import os
import asyncio

MODEL = '/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
OUTPUT_TOKENS = 64
RUNS = 3
WARMUP = 1

TEST_PROMPTS = {
    "256tok": "The quick brown fox jumps over the lazy dog. " * 40,
    "512tok": "The quick brown fox jumps over the lazy dog. " * 80,
    "1024tok": "The quick brown fox jumps over the lazy dog. " * 160,
    "2048tok": "The quick brown fox jumps over the lazy dog. " * 320,
}


async def benchmark_python_direct():
    """Benchmark Python TurboMind direct API with detailed timing."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Python TurboMind Direct API")
    print("=" * 60)

    from lmdeploy.turbomind import TurboMind
    from lmdeploy.tokenizer import Tokenizer
    from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig

    print("  Loading engine... ", end='', flush=True)
    load_start = time.perf_counter()
    ec = TurbomindEngineConfig(
        session_len=8192,
        max_batch_size=32,
        cache_block_seq_len=64,
        tp=1,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    tm = TurboMind(model_path=MODEL, engine_config=ec, trust_remote_code=True)
    tok = Tokenizer(MODEL, trust_remote_code=True)
    inst = tm.create_instance()
    load_ms = (time.perf_counter() - load_start) * 1000
    print(f"done ({load_ms:.0f}ms)")

    results = {}

    for ctx_label, prompt in TEST_PROMPTS.items():
        print(f"\n  --- {ctx_label} ---")

        tok_start = time.perf_counter()
        input_ids = tok.encode(prompt)
        tok_ms = (time.perf_counter() - tok_start) * 1000
        actual_ctx = len(input_ids)

        gen_cfg = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.7)

        # Warmup
        print(f"    warming up... ", end='', flush=True)
        async for out in inst.async_stream_infer(
            session_id=0, input_ids=input_ids, gen_config=gen_cfg,
            sequence_start=True, sequence_end=True,
        ):
            pass
        print("done")

        # Measured runs
        run_data = []
        for r in range(RUNS):
            engine_start = time.perf_counter()
            first_token_ms = None
            total_tokens = 0
            async for out in inst.async_stream_infer(
                session_id=r + 100, input_ids=input_ids, gen_config=gen_cfg,
                sequence_start=True, sequence_end=True,
            ):
                if out.status.value in (1, 2):
                    elapsed_ms = (time.perf_counter() - engine_start) * 1000
                    if first_token_ms is None:
                        first_token_ms = elapsed_ms
                    total_tokens += len(out.token_ids)

            total_ms = (time.perf_counter() - engine_start) * 1000
            ttft = first_token_ms or 0
            decode_ms = total_ms - ttft
            decode_tps = (total_tokens / decode_ms * 1000) if decode_ms > 0 else 0
            prefill_tps = (actual_ctx / ttft * 1000) if ttft > 0 else 0

            run_data.append({
                "ctx": actual_ctx,
                "tokenization_ms": tok_ms,
                "ttft_ms": ttft,
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
                "total_ms": total_ms,
                "decode_ms": decode_ms,
                "tokens": total_tokens,
            })
            print(f"    run={r+1}: tok={tok_ms:.2f}ms, TTFT={ttft:.1f}ms, "
                  f"prefill={prefill_tps:.0f} t/s, decode={decode_tps:.0f} t/s, "
                  f"total={total_ms:.1f}ms, tokens={total_tokens}")

        avg = {}
        for k in ["tokenization_ms", "ttft_ms", "prefill_tps", "decode_tps", "total_ms", "decode_ms"]:
            avg[k] = sum(d[k] for d in run_data) / len(run_data)
        results[ctx_label] = {"ctx_tokens": actual_ctx, "runs": run_data, "avg": avg}

    tm.close()
    return results


async def benchmark_rust_via_server():
    """Start Rust gRPC server, then make benchmark requests via gRPC client.

    Measures timing from client perspective with same methodology as Python.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK: Rust C++ Engine (via gRPC)")
    print("=" * 60)

    server_bin = '/mnt/data/lmdeploy/lmdeploy-rust-server/target/release/lmdeploy-rust-server'
    # Try finding the server binary
    server_bin = '/mnt/data/lmdeploy/lmdeploy-rust-server/target/debug/lmdeploy-rust-server'

    if not os.path.exists(server_bin):
        print(f"  Server binary not found, skipping Rust benchmark")
        return {}

    # Start server
    print(f"  Starting server... ", end='', flush=True)
    proc = subprocess.Popen(
        [server_bin, '--port', '50051', '--model', MODEL],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(5)  # Wait for server to start
    print("started (PID %d)" % proc.pid)

    # Make requests via gRPC
    results = {}
    # ... gRPC client code
    proc.terminate()
    proc.wait()
    return results


def benchmark_rust_via_binary():
    """Run the Rust benchmark binary."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Rust C++ Engine (binary)")
    print("=" * 60)

    benchmark_bin = '/mnt/data/lmdeploy/lmdeploy-rust-server/target/release/examples/benchmark'
    if not os.path.exists(benchmark_bin):
        print(f"  ERROR: benchmark binary not found")
        return {}

    ctx_lengths = [256, 512, 1024, 2048]
    cmd = [
        benchmark_bin, MODEL,
        '--output-length', str(OUTPUT_TOKENS),
        '--warmup', str(WARMUP),
        '--iterations', str(RUNS),
        '--context-lengths', *[str(c) for c in ctx_lengths],
    ]

    print(f"  Command: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  ERROR (rc={result.returncode}): {result.stderr[:500]}")
        return {}

    print(f"  Output:\n{result.stdout[-3000:]}")
    return {"raw": result.stdout}


def print_comparison(python_results, rust_output):
    """Print unified comparison table."""
    print("\n" + "=" * 80)
    print("PYTHON TURBOMIND RESULTS")
    print("=" * 80)

    print(f"\n{'Context':>10} | {'Tok(ms)':>8} | {'TTFT(ms)':>9} | "
          f"{'Prefill(t/s)':>13} | {'Decode(t/s)':>12} | {'Total(ms)':>10}")
    print("-" * 80)

    for ctx_label in sorted(python_results.keys()):
        data = python_results[ctx_label]
        avg = data["avg"]
        ctx = data["ctx_tokens"]
        print(f"{ctx:>10} | {avg['tokenization_ms']:>8.2f} | "
              f"{avg['ttft_ms']:>9.1f} | {avg['prefill_tps']:>13.0f} | "
              f"{avg['decode_tps']:>12.1f} | {avg['total_ms']:>10.1f}")

    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    for ctx_label in sorted(python_results.keys()):
        data = python_results[ctx_label]
        avg = data["avg"]
        ctx = data["ctx_tokens"]
        print(f"\n  {ctx_label} (ctx={ctx}):")
        print(f"    Tokenization: {avg['tokenization_ms']:.2f}ms")
        print(f"    Engine TTFT: {avg['ttft_ms']:.1f}ms")
        if avg['ttft_ms'] > 0:
            print(f"    Throughput: {ctx / avg['ttft_ms'] * 1000:.0f} tok/s")


async def main():
    print("Unified Benchmark: Python TurboMind vs Rust C++")
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_TOKENS}, Warmup: {WARMUP}, Runs: {RUNS}")

    python_results = await benchmark_python_direct()
    rust_results = benchmark_rust_via_binary()

    print_comparison(python_results, rust_results)

    output = {
        "python": python_results,
        "rust": rust_results,
        "config": {"model": MODEL, "output_tokens": OUTPUT_TOKENS, "warmup": WARMUP, "runs": RUNS},
    }

    out_path = "/mnt/data/lmdeploy/lmdeploy-rust-server/benchmark_unified.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
