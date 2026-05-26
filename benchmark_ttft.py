#!/usr/bin/env python3
"""
TurboMind Prefill Benchmark - TTFT only.

Measures TTFT (Time To First Token) = prefill time.
"""

import time
import asyncio
from pathlib import Path

from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig, GenerationConfig


async def main():
    # Test configurations
    test_cases = [
        ("512", 512),
        ("1024", 1024),
        ("2048", 2048),
        ("4096", 4096),
        ("8192", 8192),
    ]

    # Create engine config
    engine_config = TurbomindEngineConfig(
        model_format='awq',
        tp=1,
        cache_max_entry_count=0.8,
        cache_block_seq_len=64,
    )

    print("=" * 80)
    print("TurboMind Prefill Benchmark (TTFT)")
    print("=" * 80)

    # Load model
    print("[1/3] Loading TurboMind model...")
    model_path = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
    tm_model = TurboMind.from_pretrained(model_path, engine_config=engine_config)
    print("Model loaded successfully")
    print()

    # Get tokenizer
    tokenizer = tm_model.tokenizer

    # Warmup
    print("[2/3] Warmup (3 runs)...")
    warmup_prompt = "The quick brown fox jumps over the lazy dog. " * 100
    for i in range(3):
        input_ids = tokenizer.encode(warmup_prompt)
        model_inst = tm_model.create_instance()
        gen_config = GenerationConfig(max_new_tokens=1, temperature=0.0)
        generator = model_inst.async_stream_infer(
            session_id=i,
            input_ids=input_ids,
            gen_config=gen_config,
            sequence_start=True,
            sequence_end=True,
            stream_output=True,
        )
        async for _ in generator:
            break
        print(f"  warmup {i+1}/3... OK")
    print()

    # Measure prefill performance
    print("[3/3] Measuring prefill performance (5 runs each)...")
    print()
    print(f"{'Context':>8} | {'Tokens':>8} | {'Avg TTFT':>10} | {'Min TTFT':>10} | {'Avg TPS':>12} | {'Max TPS':>12}")
    print("-" * 80)

    results = {}

    for label, target_tokens in test_cases:
        # Generate prompt of approximately target length
        repeat_text = "The quick brown fox jumps over the lazy dog. "
        chars_per_token = 4
        target_chars = target_tokens * chars_per_token
        repeats = (target_chars // len(repeat_text)) + 1
        prompt = repeat_text * repeats

        # Tokenize to get exact token count
        input_ids = tokenizer.encode(prompt)
        actual_tokens = len(input_ids)

        run_times = []

        print(f"{label:>8} ({actual_tokens:>6} tok)... ", end='', flush=True)

        for run in range(5):
            # Create a new instance for each run
            model_inst = tm_model.create_instance()

            # Measure TTFT
            start = time.perf_counter()
            gen_config = GenerationConfig(max_new_tokens=1, temperature=0.0)
            generator = model_inst.async_stream_infer(
                session_id=run,
                input_ids=input_ids,
                gen_config=gen_config,
                sequence_start=True,
                sequence_end=True,
                stream_output=True,
            )
            # Get first token
            async for _ in generator:
                break
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            run_times.append(elapsed_ms)
            print(f"{elapsed_ms:.1f}ms ", end='', flush=True)

        if run_times:
            avg_ms = sum(run_times) / len(run_times)
            min_ms = min(run_times)
            avg_tps = actual_tokens / avg_ms * 1000.0
            max_tps = actual_tokens / min_ms * 1000.0

            results[label] = {
                'tokens': actual_tokens,
                'avg_ms': avg_ms,
                'min_ms': min_ms,
                'avg_tps': avg_tps,
                'max_tps': max_tps,
            }

            print(f"| Avg {avg_tps:>10.1f} tok/s, Max {max_tps:>10.1f} tok/s")

    # Summary
    print()
    print("=" * 80)
    print("Summary - TurboMind Prefill Performance")
    print("=" * 80)
    print()
    print(f"{'Context':>8} | {'Tokens':>8} | {'Avg TTFT':>10} | {'Avg TPS':>12} | {'Max TPS':>12}")
    print("-" * 80)
    for label in ["512", "1024", "2048", "4096", "8192"]:
        if label in results:
            r = results[label]
            print(f"{label:>8} | {r['tokens']:>8} | {r['avg_ms']:>10.1f}ms | {r['avg_tps']:>12.1f} | {r['max_tps']:>12.1f}")


if __name__ == '__main__':
    asyncio.run(main())
