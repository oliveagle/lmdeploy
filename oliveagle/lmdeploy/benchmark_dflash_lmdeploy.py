#!/usr/bin/env python3
"""
DFlash Benchmark for lmdeploy PyTorch backend.

Tests DFlash speculative decoding performance on Qwen3-8B.

Usage:
    python3 benchmark_dflash_lmdeploy.py --target-model /path/to/qwen3-8b-hf \\
                                       --draft-model /path/to/qwen3-8b-dflash \\
                                       --num-spec-tokens 8
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# Ensure lmdeploy is in path
LMPATH = '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy'
sys.path.insert(0, LMPATH)

# Test prompts (similar to llama.cpp test)
TEST_PROMPTS = [
    ("Thinking mode - short output",
     "Write a Python quicksort function.",
     128),
    ("Thinking mode - simple Q&A",
     "What is 2+2?",
     128),
    ("No-thinking mode - long output",
     "Explain the difference between TCP and UDP in networking. Include details about handshakes, reliability, and use cases.",
     256),
]

TARGET_MODEL_DEFAULT = "/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3-4B"
DRAFT_MODEL_DEFAULT = "/mnt/eaget-4tb/modelscope_models/z-lab/Qwen3-4B-DFlash-b16"


def run_test_simple(model_path, prompt, max_tokens, temperature=0.7, top_p=0.9):
    """Run a simple test without speculative decoding."""
    print(f"\n{'='*60}")
    print(f"Test: {prompt[:60]}...")
    print(f"{'='*60}")

    try:
        from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

        pipe = pipeline(
            model_path,
            backend_config=PytorchEngineConfig(
                dtype='bfloat16',
                cache_max_entry_count=0.85,
            ),
            log_level='ERROR',
        )

        print(f"Model loaded, starting inference...")
        start = time.time()
        output = pipe(
            prompt,
            GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            ),
        )
        elapsed = time.time() - start

        num_tokens = len(output.token_ids)
        tps = num_tokens / elapsed

        print(f"Generated {num_tokens} tokens in {elapsed:.2f}s")
        print(f"Speed: {tps:.2f} tokens/s")
        print(f"Output preview: {output.text[:150]}...")

        return {
            'tokens': num_tokens,
            'time': elapsed,
            'tps': tps,
            'output': output.text,
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_test_dflash(target_model, draft_model, prompt, max_tokens, num_spec_tokens, temperature=0.7, top_p=0.9):
    """Run DFlash speculative decoding test."""
    print(f"\n{'='*60}")
    print(f"DFlash Test ({num_spec_tokens} spec tokens): {prompt[:60]}...")
    print(f"{'='*60}")

    try:
        from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
        from lmdeploy.messages import SpeculativeConfig

        print(f"Loading target model: {target_model}")
        print(f"Loading draft model: {draft_model}")
        print(f"Num speculative tokens: {num_spec_tokens}")

        pipe = pipeline(
            target_model,
            speculative_config=SpeculativeConfig(
                method='dflash',
                model=draft_model,
                num_speculative_tokens=num_spec_tokens,
            ),
            backend_config=PytorchEngineConfig(
                dtype='bfloat16',
                cache_max_entry_count=0.85,
            ),
            log_level='INFO',
        )

        print(f"Models loaded, starting inference...")
        start = time.time()
        output = pipe(
            prompt,
            GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            ),
        )
        elapsed = time.time() - start

        num_tokens = len(output.token_ids)
        tps = num_tokens / elapsed

        print(f"Generated {num_tokens} tokens in {elapsed:.2f}s")
        print(f"Speed: {tps:.2f} tokens/s")
        print(f"Output preview: {output.text[:150]}...")

        return {
            'tokens': num_tokens,
            'time': elapsed,
            'tps': tps,
            'output': output.text,
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='DFlash benchmark for lmdeploy')
    parser.add_argument('--target-model', type=str, default=TARGET_MODEL_DEFAULT,
                        help='Path to target model (Qwen3-8B-HF)')
    parser.add_argument('--draft-model', type=str, default=DRAFT_MODEL_DEFAULT,
                        help='Path to DFlash draft model')
    parser.add_argument('--num-spec-tokens', type=int, default=8,
                        help='Number of speculative tokens')
    parser.add_argument('--max-tokens', type=int, default=128,
                        help='Max tokens to generate')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline test')
    parser.add_argument('--skip-dflash', action='store_true',
                        help='Skip DFlash test')
    parser.add_argument('--output', type=str, default='dflash_lmdeploy_benchmark.json',
                        help='Output JSON file')

    args = parser.parse_args()

    # Verify models exist
    if not os.path.exists(args.target_model):
        print(f"ERROR: Target model not found: {args.target_model}")
        return 1

    if not os.path.exists(args.draft_model):
        print(f"ERROR: Draft model not found: {args.draft_model}")
        return 1

    results = {
        'target_model': args.target_model,
        'draft_model': args.draft_model,
        'num_spec_tokens': args.num_spec_tokens,
        'tests': [],
    }

    print("="*60)
    print("DFlash Benchmark for lmdeploy")
    print("="*60)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Num speculative tokens: {args.num_spec_tokens}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*60)

    for i, (test_name, prompt, max_tokens) in enumerate(TEST_PROMPTS):
        print(f"\n\n{'#'*60}")
        print(f"Test {i+1}/{len(TEST_PROMPTS)}: {test_name}")
        print(f"{'#'*60}")

        test_result = {
            'name': test_name,
            'prompt': prompt,
            'max_tokens': max_tokens,
        }

        # Baseline
        if not args.skip_baseline:
            print(f"\n--- Running baseline (no speculative) ---")
            baseline = run_test_simple(args.target_model, prompt, min(max_tokens, args.max_tokens))
            test_result['baseline'] = baseline
            if baseline:
                print(f"  → {baseline['tps']:.2f} tokens/s")

        # DFlash
        if not args.skip_dflash:
            print(f"\n--- Running DFlash (speculative) ---")
            dflash = run_test_dflash(
                args.target_model,
                args.draft_model,
                prompt,
                min(max_tokens, args.max_tokens),
                args.num_spec_tokens,
            )
            test_result['dflash'] = dflash
            if dflash:
                print(f"  → {dflash['tps']:.2f} tokens/s")

                # Calculate speedup
                if baseline and baseline.get('tps'):
                    speedup = dflash['tps'] / baseline['tps']
                    test_result['speedup'] = speedup
                    print(f"  → Speedup: {speedup:.2f}x")

        results['tests'].append(test_result)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")

    # Summary
    print("\nSummary:")
    baseline_tpss = [t['baseline']['tps'] for t in results['tests'] if t.get('baseline') and t['baseline'].get('tps')]
    dflash_tpss = [t['dflash']['tps'] for t in results['tests'] if t.get('dflash') and t['dflash'].get('tps')]

    if baseline_tpss:
        avg_baseline = sum(baseline_tpss) / len(baseline_tpss)
        print(f"  Avg baseline: {avg_baseline:.2f} tokens/s")
    if dflash_tpss:
        avg_dflash = sum(dflash_tpss) / len(dflash_tpss)
        print(f"  Avg DFlash:   {avg_dflash:.2f} tokens/s")
        if baseline_tpss:
            print(f"  Avg speedup:  {avg_dflash/avg_baseline:.2f}x")

    return 0


if __name__ == '__main__':
    sys.exit(main())
