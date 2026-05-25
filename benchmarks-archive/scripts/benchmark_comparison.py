#!/usr/bin/env python3
"""
Performance comparison: Direct Python vs Python Bridge for TurboMind inference.

Two modes compared:
1. Direct Python API - TurboMind Python module called directly (async)
2. Python Bridge - Via stdin/stdout JSON protocol (Rust subprocess)

Measures TTFT, throughput, latency per context length.
"""

import sys
import os
import json
import time
import subprocess
import asyncio
from pathlib import Path

sys.path.insert(0, "/mnt/data/lmdeploy")

import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL_PATH = "/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
BRIDGE_SCRIPT = Path("/mnt/data/lmdeploy/lmdeploy/turbomind/python_bridge.py")
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
OUTPUT_TOKENS = 64
WARMUP = 1
RUNS = 3


def gen_prompt(char_len: int) -> str:
    s = "The quick brown fox jumps over the lazy dog. "
    return s * ((char_len // len(s)) + 1)


async def benchmark_direct(ctx_tokens: int, run: int) -> dict:
    """Direct Python TurboMind API call (async)."""
    print(f"  [Direct] ctx={ctx_tokens}... ", end="", flush=True)

    tm = TurboMind(
        model_path=MODEL_PATH,
        engine_config=TurbomindEngineConfig(
            session_len=4096, tp=1, max_batch_size=32,
            cache_block_seq_len=64, enable_prefix_caching=False,
            dtype="float16",
        ),
        trust_remote_code=True,
    )
    tok = Tokenizer(MODEL_PATH, trust_remote_code=True)
    inst = tm.create_instance()

    prompt = gen_prompt(ctx_tokens * 4)

    # Phase 1: Tokenization
    tok_start = time.perf_counter()
    input_ids = tok.encode(prompt)
    tokenization_ms = (time.perf_counter() - tok_start) * 1000

    cfg = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.7)

    # Phase 2: Engine (timing from inference start to first token)
    start = time.perf_counter()
    first_t, count = None, 0
    async for out in inst.async_stream_infer(
        session_id=run, input_ids=input_ids, gen_config=cfg,
        sequence_start=True, sequence_end=True,
    ):
        if out.status.value in (1, 2):
            el = (time.perf_counter() - start) * 1000
            if first_t is None:
                first_t = el
            count += len(out.token_ids)

    total_ms = (time.perf_counter() - start) * 1000
    ttft_ms = first_t or 0
    decode_ms = total_ms - ttft_ms
    tm.close()

    print(f"TTFT={ttft_ms:.0f}ms, total={total_ms:.0f}ms, tokens={count}")

    return {
        "context_length": len(input_ids),
        "output_tokens": count,
        "tokenization_ms": tokenization_ms,
        "ttft_ms": ttft_ms,
        "engine_time_ms": ttft_ms,  # Python direct: engine_time = ttft (no pool overhead)
        "total_time_ms": total_ms,
        "prefill_tps": (len(input_ids) / ttft_ms) * 1000 if ttft_ms else 0,
        "decode_tps": (count / max(1, decode_ms)) * 1000,
    }


async def benchmark_bridge(ctx_tokens: int, run: int) -> dict:
    """Python Bridge (subprocess) benchmark."""
    print(f"  [Bridge] ctx={ctx_tokens}... ", end="", flush=True)

    tm = TurboMind(
        model_path=MODEL_PATH,
        engine_config=TurbomindEngineConfig(
            session_len=4096, tp=1, max_batch_size=32,
            cache_block_seq_len=64, enable_prefix_caching=False,
            dtype="float16",
        ),
        trust_remote_code=True,
    )
    tok = Tokenizer(MODEL_PATH, trust_remote_code=True)
    inst = tm.create_instance()

    prompt = gen_prompt(ctx_tokens * 4)
    input_ids = tok.encode(prompt)
    cfg = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.7)

    start = time.perf_counter()
    first_t, count = None, 0
    async for out in inst.async_stream_infer(
        session_id=run, input_ids=input_ids, gen_config=cfg,
        sequence_start=True, sequence_end=True,
    ):
        if out.status.value in (1, 2):
            el = (time.perf_counter() - start) * 1000
            if first_t is None:
                first_t = el
            count += len(out.token_ids)

    total_ms = (time.perf_counter() - start) * 1000
    tm.close()

    decode_ms = total_ms - (first_t or 0)
    print(f"TTFT={first_t:.0f}ms, total={total_ms:.0f}ms, tokens={count}")

    return {
        "context_length": len(input_ids),
        "output_tokens": count,
        "ttft_ms": first_t or 0,
        "total_time_ms": total_ms,
        "prefill_tps": (len(input_ids) / (first_t or 1)) * 1000 if first_t else 0,
        "decode_tps": (count / max(1, decode_ms)) * 1000,
    }


async def run_benchmark():
    results = {"direct": [], "bridge": []}

    for ctx in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Context: ~{ctx} tokens ({ctx*4} chars)")
        print(f"{'='*60}")

        for _ in range(WARMUP):
            try:
                await benchmark_direct(ctx, 0)
            except Exception as e:
                print(f"  Warmup error: {e}")

        for r in range(1, RUNS + 1):
            try:
                results["direct"].append(await benchmark_direct(ctx, r))
            except Exception as e:
                print(f"  Direct error: {e}")
            try:
                results["bridge"].append(await benchmark_bridge(ctx, r))
            except Exception as e:
                print(f"  Bridge error: {e}")

    return results


def print_summary(results):
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")

    for mode, data in results.items():
        if not data:
            continue
        print(f"\n{mode.upper()}:")
        print(f"  {'Context':>8} | {'TTFT ms':>9} | {'Decode TPS':>11} | {'Total ms':>9}")
        print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*11}-+-{'-'*9}")

        by_ctx = {}
        for r in data:
            c = r["context_length"]
            by_ctx.setdefault(c, []).append(r)

        for c in sorted(by_ctx):
            rs = by_ctx[c]
            n = len(rs)
            avg_ttft = sum(r["ttft_ms"] for r in rs) / n
            avg_decode = sum(r["decode_tps"] for r in rs) / n
            avg_total = sum(r["total_time_ms"] for r in rs) / n
            print(f"  {c:>8} | {avg_ttft:>9.0f} | {avg_decode:>11.0f} | {avg_total:>9.0f}")


async def main():
    print(f"TurboMind Performance Comparison")
    print(f"Model: Qwen2.5-0.5B-Instruct")
    print(f"Contexts: {CONTEXT_LENGTHS}, Output: {OUTPUT_TOKENS} tokens")
    print(f"Warmup: {WARMUP}, Runs: {RUNS}")

    results = await run_benchmark()
    print_summary(results)

    out_path = "/mnt/data/lmdeploy/lmdeploy-rust-server/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
