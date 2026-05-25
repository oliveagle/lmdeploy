#!/usr/bin/env python3
"""Benchmark Qwen3.6-27B-AWQ on LMDeploy main branch using PyTorch backend."""

import os
import time
import json
from datetime import datetime

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

def main():
    model_path = '/mnt/data/models/modelscope_models/Qwen3___6-27B-AWQ'
    model_name = 'Qwen3.6-27B-AWQ'

    import subprocess
    gpu_info = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'],
        capture_output=True, text=True
    ).stdout.strip()

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"GPU: {gpu_info}")
    print(f"Backend: PyTorch (AWQ workaround)")
    print(f"{'='*70}")

    print("\nInitializing pipeline...", end=' ', flush=True)
    try:
        pipe = pipeline(
            model_path,
            backend='pytorch',
            backend_config=PytorchEngineConfig(
                cache_max_entry_count=0.8,
                block_size=16,
                max_batch_size=32,
            )
        )
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    scenarios = [
        {'name': '4K-512', 'context_len': 4096, 'output_len': 512,
         'prompt': 'Write a Python quicksort function with type hints.'},
        {'name': '8K-512', 'context_len': 8192, 'output_len': 512,
         'prompt': 'Write a Python quicksort function. ' + 'Explain the algorithm step by step. ' * 50},
        {'name': '16K-512', 'context_len': 16384, 'output_len': 512,
         'prompt': 'Write a Python quicksort function. ' + 'Explain the algorithm step by step. ' * 120},
    ]

    results = []
    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['name']}...", end=' ', flush=True)
        try:
            start = time.time()
            output = pipe([scenario['prompt']], gen_config=GenerationConfig(max_new_tokens=scenario['output_len']))
            end = time.time()

            elapsed = end - start
            tokens = len(output[0].token_ids) if output else 0
            tps = tokens / elapsed if elapsed > 0 else 0

            print(f"{tps:.2f} tokens/s ({tokens} tokens in {elapsed:.2f}s)")
            results.append({
                'model': model_name, 'scenario': scenario['name'],
                'context_len': scenario['context_len'], 'output_len': scenario['output_len'],
                'tokens_generated': tokens, 'time': elapsed, 'tokens_per_sec': tps, 'status': 'success',
            })
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model': model_name, 'scenario': scenario['name'],
                'error': str(e), 'status': 'error',
            })

    # Save results
    output_file = f'benchmark_27b_pytorch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump({'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'gpu': gpu_info,
                    'model': model_name, 'backend': 'pytorch', 'results': results}, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'Scenario':<12} {'Speed (t/s)':<15} {'Status':<10}")
    print("-" * 40)
    for r in results:
        if r['status'] == 'success':
            print(f"{r['scenario']:<12} {r['tokens_per_sec']:<15.2f} {r['status']:<10}")
        else:
            print(f"{r['scenario']:<12} {'ERROR':<15} {r['status']:<10}")
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()