#!/usr/bin/env python3
"""
Unified benchmark: Python TurboMind vs Rust C++ with phase-separated timing.

Measures each phase independently:
- tokenization_time_ms: encode prompt → token IDs
- pool_acquire_time_ms: acquire inference slot (Rust only)
- engine_time_ms: pure C++/CUDA engine time (TTFT - overhead)
- ttft_ms: total time to first token (end-to-end)
- prefill_tps: input_tokens / (engine_time_ms / 1000)
- decode_tps: output_tokens / (decode_time_ms / 1000)

This allows apple-to-apple comparison between Python and Rust backends
by isolating the engine performance from Python-side overhead.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = os.environ.get(
    "MODEL_PATH",
    "/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
)
OUTPUT_TOKENS = 64
WARMUP = 1
RUNS = 3

TEST_PROMPTS = {
    "256tok": "The quick brown fox jumps over the lazy dog. " * 40,
    "512tok": "The quick brown fox jumps over the lazy dog. " * 80,
    "1024tok": "The quick brown fox jumps over the lazy dog. " * 160,
    "2048tok": "The quick brown fox jumps over the lazy dog. " * 320,
}


def gen_prompt(char_len: int) -> str:
    s = "The quick brown fox jumps over the lazy dog. "
    return s * ((char_len // len(s)) + 1)


async def benchmark_python_direct() -> dict:
    """Benchmark Python TurboMind with phase-separated timing."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Python TurboMind Direct")
    print("=" * 60)

    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=8192,
            max_batch_size=32,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            dtype="bfloat16",
        ),
        trust_remote_code=True,
    )
    tok = Tokenizer(MODEL, trust_remote_code=True)
    inst = tm.create_instance()

    results = {}
    for ctx_label, prompt in TEST_PROMPTS.items():
        print(f"\n  --- {ctx_label} ---")

        # Phase 1: Tokenization
        tok_start = time.perf_counter()
        input_ids = tok.encode(prompt)
        tokenization_ms = (time.perf_counter() - tok_start) * 1000
        actual_ctx = len(input_ids)

        gen_cfg = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.7)

        # Warmup
        print(f"    warming up... ", end="", flush=True)
        async for out in inst.async_stream_infer(
            session_id=0, input_ids=input_ids, gen_config=gen_cfg,
            sequence_start=True, sequence_end=True,
        ):
            pass
        print("done")

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
                "engine_time_ms": ttft,  # Python direct: engine_time = ttft (no pool overhead)
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
                "total_ms": total_ms,
                "decode_ms": decode_ms,
                "tokens": total_tokens,
            })
            print(f"    run={r+1}: tok={tokenization_ms:.2f}ms, TTFT={ttft:.1f}ms, "
                  f"prefill={prefill_tps:.0f} t/s, decode={decode_tps:.0f} t/s, "
                  f"total={total_ms:.1f}ms, tokens={total_tokens}")

        avg = {}
        for k in ["tokenization_ms", "ttft_ms", "engine_time_ms", "prefill_tps",
                   "decode_tps", "total_ms", "decode_ms"]:
            avg[k] = sum(d[k] for d in run_data) / len(run_data)
        results[ctx_label] = {"ctx_tokens": actual_ctx, "runs": run_data, "avg": avg}

    tm.close()
    return results


def benchmark_rust_via_binary() -> dict:
    """Run the Rust benchmark binary and parse its JSON output."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Rust C++ Engine (binary)")
    print("=" * 60)

    benchmark_bin = str(Path(__file__).parent.parent / "target" / "release" / "examples" / "benchmark")
    if not os.path.exists(benchmark_bin):
        benchmark_bin = str(Path(__file__).parent.parent / "target" / "debug" / "examples" / "benchmark")

    if not os.path.exists(benchmark_bin):
        print("  WARNING: Rust benchmark binary not found. Build with: cargo build --example benchmark")
        return {}

    ctx_lengths = [256, 512, 1024, 2048]
    cmd = [
        benchmark_bin, MODEL,
        "--output-length", str(OUTPUT_TOKENS),
        "--warmup", str(WARMUP),
        "--iterations", str(RUNS),
        "--context-lengths", *[str(c) for c in ctx_lengths],
    ]

    print(f"  Running: {benchmark_bin} --model {MODEL} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  ERROR (rc={result.returncode}): {result.stderr[-500:]}")
        return {}

    print(f"  Output (last 1500 chars):\n{result.stdout[-1500:]}")
    return {"raw_output": result.stdout, "raw_stderr": result.stderr[-500:]}


def print_comparison(python_results: dict, rust_results: dict):
    """Print unified comparison table with phase-separated timing."""
    print("\n" + "=" * 90)
    print("UNIFIED BENCHMARK COMPARISON: Python TurboMind vs Rust C++")
    print("=" * 90)

    # Python results
    print(f"\n{'PYTHON TURBOMIND':^60}")
    print(f"  {'Context':>8} | {'Tok(ms)':>8} | {'Engine(ms)':>10} | "
          f"{'TTFT(ms)':>9} | {'Prefill(t/s)':>13} | {'Decode(t/s)':>12}")
    print("  " + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 10 + "-+-" + "-" * 9
          + "-+-" + "-" * 13 + "-+-" + "-" * 12)

    for ctx_label in sorted(python_results.keys()):
        data = python_results[ctx_label]
        avg = data["avg"]
        ctx = data["ctx_tokens"]
        print(f"  {ctx:>8} | {avg['tokenization_ms']:>8.2f} | "
              f"{avg['engine_time_ms']:>10.1f} | {avg['ttft_ms']:>9.1f} | "
              f"{avg['prefill_tps']:>13.0f} | {avg['decode_tps']:>12.1f}")

    # Key findings
    print(f"\n{'=' * 60}")
    print("KEY FINDINGS")
    print(f"{'=' * 60}")
    for ctx_label in sorted(python_results.keys()):
        data = python_results[ctx_label]
        avg = data["avg"]
        ctx = data["ctx_tokens"]
        print(f"\n  {ctx_label} (ctx={ctx}):")
        print(f"    Tokenization: {avg['tokenization_ms']:.2f}ms")
        print(f"    Engine time (TTFT): {avg['ttft_ms']:.1f}ms")
        if avg["ttft_ms"] > 0:
            engine_tps = ctx / avg["ttft_ms"] * 1000
            print(f"    Engine throughput: {engine_tps:.0f} tok/s")

    # Rust output summary
    if rust_results.get("raw_output"):
        print(f"\n{'=' * 60}")
        print("RUST C++ ENGINE OUTPUT")
        print(f"{'=' * 60}")
        print(rust_results.get("raw_output", "")[-2000:])


async def main():
    print("Unified Benchmark: Python TurboMind vs Rust C++")
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_TOKENS}, Warmup: {WARMUP}, Runs: {RUNS}")
    print(f"Context labels: {list(TEST_PROMPTS.keys())}")

    python_results = await benchmark_python_direct()
    rust_results = benchmark_rust_via_binary()

    print_comparison(python_results, rust_results)

    # Save combined results
    output = {
        "python_direct": python_results,
        "rust_binary": rust_results,
        "config": {
            "model": MODEL,
            "output_tokens": OUTPUT_TOKENS,
            "warmup": WARMUP,
            "runs": RUNS,
            "test_prompts": dict(TEST_PROMPTS),
        },
    }

    out_path = Path(__file__).parent.parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
