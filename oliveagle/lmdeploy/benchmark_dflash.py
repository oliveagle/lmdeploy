#!/usr/bin/env python3
"""DFlash benchmark script for lmdeploy.

This script benchmarks the DFlash speculative decoding implementation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add lmdeploy to path
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

try:
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
    from lmdeploy.pytorch.config import SpecDecodeConfig
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Error importing lmdeploy: {e}")
    print("Please ensure lmdeploy is properly installed.")
    sys.exit(1)


def benchmark_baseline(model_path: str, prompt: str, max_tokens: int = 128):
    """Benchmark baseline inference without speculative decoding."""
    print("\n" + "=" * 60)
    print("Baseline (No Speculative Decoding)")
    print("=" * 60)

    try:
        pipe = pipeline(
            model_path,
            backend_config=PytorchEngineConfig(
                dtype='bfloat16',
                gpu_memory_utilization=0.85,
            ),
            log_level='ERROR',
        )

        start = time.time()
        output = pipe(
            prompt,
            GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            ),
        )
        elapsed = time.time() - start

        num_tokens = len(output[0].token_ids)
        tps = num_tokens / elapsed

        print(f"Generated {num_tokens} tokens in {elapsed:.2f}s")
        print(f"Speed: {tps:.2f} tokens/s")
        print(f"Output: {output[0].text[:200]}...")

        return {
            'tokens': num_tokens,
            'time': elapsed,
            'tps': tps,
            'output': output[0].text,
        }
    except Exception as e:
        print(f"Baseline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_dflash(target_model: str, draft_model: str, prompt: str, max_tokens: int = 128, num_spec_tokens: int = 8):
    """Benchmark DFlash speculative decoding."""
    print("\n" + "=" * 60)
    print(f"DFlash Speculative Decoding ({num_spec_tokens} speculative tokens)")
    print("=" * 60)

    try:
        pipe = pipeline(
            target_model,
            speculative_config=SpecDecodeConfig(
                method='dflash',
                model=draft_model,
                num_speculative_tokens=num_spec_tokens,
            ),
            backend_config=PytorchEngineConfig(
                dtype='bfloat16',
                gpu_memory_utilization=0.85,
            ),
            log_level='INFO',  # Use INFO to see DFlash logs
        )

        start = time.time()
        output = pipe(
            prompt,
            GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            ),
        )
        elapsed = time.time() - start

        num_tokens = len(output[0].token_ids)
        tps = num_tokens / elapsed

        print(f"Generated {num_tokens} tokens in {elapsed:.2f}s")
        print(f"Speed: {tps:.2f} tokens/s")
        print(f"Output: {output[0].text[:200]}...")

        return {
            'tokens': num_tokens,
            'time': elapsed,
            'tps': tps,
            'output': output[0].text,
        }
    except Exception as e:
        print(f"DFlash failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='DFlash benchmark for lmdeploy')
    parser.add_argument('--target-model', type=str,
                        default='/mnt/eaget-4tb/models/Qwen3-8B-HF',
                        help='Path to target model')
    parser.add_argument('--draft-model', type=str,
                        default='/mnt/eaget-4tb/models/Qwen3-8B-DFlash',
                        help='Path to DFlash draft model')
    parser.add_argument('--num-spec-tokens', type=int, default=8,
                        help='Number of speculative tokens')
    parser.add_argument('--max-tokens', type=int, default=128,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline benchmark')
    parser.add_argument('--skip-dflash', action='store_true',
                        help='Skip DFlash benchmark')

    args = parser.parse_args()

    # Test prompts
    prompts = [
        "Write a Python quicksort function.",
        "Explain the difference between TCP and UDP in networking.",
        "What is the capital of France? Describe it briefly.",
    ]

    results = {
        'target_model': args.target_model,
        'draft_model': args.draft_model,
        'num_spec_tokens': args.num_spec_tokens,
        'prompts': prompts,
    }

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Test Prompt {i+1}/{len(prompts)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:100]}...")

        # Baseline
        if not args.skip_baseline:
            baseline_result = benchmark_baseline(args.target_model, prompt, args.max_tokens)
            if baseline_result:
                results[f'prompt_{i}_baseline'] = baseline_result

        # DFlash
        if not args.skip_dflash:
            dflash_result = benchmark_dflash(
                args.target_model, args.draft_model, prompt, args.max_tokens, args.num_spec_tokens
            )
            if dflash_result:
                results[f'prompt_{i}_dflash'] = dflash_result

                # Calculate speedup if baseline available
                baseline_key = f'prompt_{i}_baseline'
                if baseline_key in results and results[baseline_key]:
                    baseline_tps = results[baseline_key]['tps']
                    dflash_tps = dflash_result['tps']
                    speedup = dflash_tps / baseline_tps
                    print(f"\nSpeedup: {speedup:.2f}x")
                    results[f'prompt_{i}_speedup'] = speedup

    # Save results
    output_file = '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy/dflash_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    for i in range(len(prompts)):
        baseline_key = f'prompt_{i}_baseline'
        dflash_key = f'prompt_{i}_dflash'
        speedup_key = f'prompt_{i}_speedup'

        if baseline_key in results and dflash_key in results:
            print(f"Prompt {i+1}:")
            print(f"  Baseline: {results[baseline_key]['tps']:.2f} tokens/s")
            print(f"  DFlash:   {results[dflash_key]['tps']:.2f} tokens/s")
            if speedup_key in results:
                print(f"  Speedup:  {results[speedup_key]:.2f}x")


if __name__ == '__main__':
    main()
