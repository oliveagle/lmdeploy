#!/usr/bin/env python3
"""Benchmark Qwen3.6-35B-A3B-AWQ prefill performance - measuring FIRST TOKEN latency."""

import os
import time
import json
from datetime import datetime

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

def main():
    model_path = '/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ'
    model_name = 'Qwen3.6-35B-A3B-AWQ'

    import subprocess
    gpu_info = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'],
        capture_output=True, text=True
    ).stdout.strip()

    print(f"\n{'='*70}")
    print(f"LMDeploy Main Branch - Prefill Benchmark (First Token Latency)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"GPU: {gpu_info}")
    print(f"Backend: TurboMind")
    print(f"{'='*70}")

    print("\nInitializing pipeline...", end=' ', flush=True)
    try:
        pipe = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(
                tp=1,
                cache_max_entry_count=0.85,
                model_format='awq',
            )
        )
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    tokenizer = pipe.async_engine.tokenizer.model.model

    # Prefill scenarios
    scenarios = [
        {'name': '1K prefill', 'prompt_tokens': 1024},
        {'name': '4K prefill', 'prompt_tokens': 4096},
        {'name': '8K prefill', 'prompt_tokens': 8192},
        {'name': '16K prefill', 'prompt_tokens': 16384},
        {'name': '32K prefill', 'prompt_tokens': 32768},
    ]

    results = []
    base_prompt = "This is a benchmark test. " + "The quick brown fox jumps over the lazy dog. " * 100

    for scenario in scenarios:
        name = scenario['name']
        target_tokens = scenario['prompt_tokens']

        print(f"\n{'='*70}")
        print(f"Scenario: {name}")

        # Create prompt with approximately target_tokens tokens
        base_tokens = len(tokenizer.encode(base_prompt))
        prompt = base_prompt * (target_tokens // base_tokens + 1)

        # Trim to exact target tokens
        encoded = tokenizer.encode(prompt)[:target_tokens]
        prompt = tokenizer.decode(encoded)

        # Actual token count
        actual_tokens = len(tokenizer.encode(prompt))
        print(f"Actual prompt tokens: {actual_tokens}")

        # Warmup (skip for first scenario to avoid cache effects)
        if scenario == scenarios[0]:
            print("Warming up...")
            _ = pipe(["Warmup"], gen_config=GenerationConfig(max_new_tokens=10))
            print("Warmup complete")

        try:
            # Measure first token latency (prefill time)
            start = time.time()
            for output in pipe.stream_infer([prompt], gen_config=GenerationConfig(max_new_tokens=512)):
                first_token_time = time.time()
                # output is a Response object, check if it has tokens
                if output.token_ids:
                    break

            prefill_time = first_token_time - start

            # Count tokens generated so far
            tokens_so_far = len(output.token_ids) if output else 0

            # Measure decode speed with remaining tokens
            decode_start = time.time()
            decode_end = time.time()
            decode_time = decode_end - decode_start
            decode_tokens = 0  # Can't measure decode speed this way
            decode_tps = 0

            # Measure decode speed separately
            decode_output = pipe([prompt], gen_config=GenerationConfig(max_new_tokens=256))
            decode_end = time.time() - decode_start
            decode_time = decode_end
            decode_tokens = len(decode_output[0].token_ids) if decode_output else 0
            decode_tps = decode_tokens / decode_time if decode_time > 0 else 0

            # Prefill speed = tokens / prefill_time
            prefill_tps = actual_tokens / prefill_time if prefill_time > 0 else 0

            print(f"First token time: {prefill_time:.4f}s")
            print(f"Prefill speed: {prefill_tps:.2f} tokens/s")
            print(f"Decode time: {decode_time:.4f}s ({decode_tokens} tokens)")
            print(f"Decode speed: {decode_tps:.2f} tokens/s")

            results.append({
                'scenario': name,
                'target_prompt_tokens': target_tokens,
                'actual_prompt_tokens': actual_tokens,
                'prefill_time': prefill_time,
                'prefill_tokens_per_sec': prefill_tps,
                'decode_time': decode_time,
                'decode_tokens': decode_tokens,
                'decode_tokens_per_sec': decode_tps,
                'status': 'success',
            })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scenario': name,
                'target_prompt_tokens': target_tokens,
                'error': str(e),
                'status': 'error',
            })

    # Save results
    output_file = f'prefill_first_token_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    final_results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': gpu_info,
        'model': model_name,
        'backend': 'turbomind',
        'results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("PREFILL BENCHMARK SUMMARY (First Token Latency)")
    print(f"{'='*70}")
    print(f"{'Scenario':<15} {'Tokens':<10} {'Prefill (s)':<15} {'Prefill t/s':<15} {'Status':<10}")
    print("-" * 65)
    for r in results:
        if r['status'] == 'success':
            print(f"{r['scenario']:<15} {r['actual_prompt_tokens']:<10} {r['prefill_time']:<15.4f} {r['prefill_tokens_per_sec']:<15.2f} {r['status']:<10}")
        else:
            print(f"{r['scenario']:<15} {r.get('actual_prompt_tokens', 'N/A'):<10} {'ERROR':<15} {'ERROR':<15} {r['status']:<10}")

    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()