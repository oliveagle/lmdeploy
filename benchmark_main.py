#!/usr/bin/env python3
"""Benchmark LMDeploy main branch on Qwen AWQ models."""

import os
import time
import json
from datetime import datetime

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

# Models to test
MODELS = [
    {
        'name': 'Qwen3.6-35B-A3B-AWQ',
        'path': '/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ',
        'gpu_memory_utilization': 0.85,
    },
    {
        'name': 'Qwen3.6-27B-AWQ',
        'path': '/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-27B-AWQ',
        'gpu_memory_utilization': 0.85,
    },
]

# Test scenarios
SCENARIOS = [
    {
        'name': '4K-512',
        'context_len': 4096,
        'output_len': 512,
        'prompt': 'Write a Python quicksort function with type hints that sorts a list of numbers in ascending order.',
    },
    {
        'name': '8K-512',
        'context_len': 8192,
        'output_len': 512,
        'prompt': 'Write a Python quicksort function with type hints. ' + 'Explain the algorithm step by step. ' * 50,
    },
]

def run_benchmark(model_cfg: dict, scenarios: list):
    """Run benchmark for a single model."""
    import subprocess

    results = []

    # Get GPU info
    gpu_info = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'],
        capture_output=True, text=True
    ).stdout.strip()

    model_name = model_cfg['name']
    model_path = model_cfg['path']

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"GPU: {gpu_info}")
    print(f"{'='*70}")

    # Initialize pipeline
    print("\nInitializing pipeline...", end=' ', flush=True)
    pipe = None
    try:
        pipe = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(
                gpu_memory_utilization=model_cfg['gpu_memory_utilization']
            )
        )
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        results.append({
            'model': model_name,
            'error': f'Failed to initialize pipeline: {e}',
            'status': 'error',
        })
        return results

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['name']}...", end=' ', flush=True)

        prompt = scenario['prompt']
        output_len = scenario['output_len']

        try:
            # Run inference
            start = time.time()
            output = pipe([prompt], gen_config=GenerationConfig(max_new_tokens=output_len))
            end = time.time()

            elapsed = end - start
            tokens = len(output[0].token_ids) if output else 0
            tps = tokens / elapsed if elapsed > 0 else 0

            print(f"{tps:.2f} tokens/s ({tokens} tokens in {elapsed:.2f}s)")

            results.append({
                'model': model_name,
                'scenario': scenario['name'],
                'context_len': scenario['context_len'],
                'output_len': output_len,
                'tokens_generated': tokens,
                'time': elapsed,
                'tokens_per_sec': tps,
                'status': 'success',
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'model': model_name,
                'scenario': scenario['name'],
                'context_len': scenario['context_len'],
                'output_len': output_len,
                'error': str(e),
                'status': 'error',
            })

        time.sleep(0.5)

    return results


def main():
    print("=" * 70)
    print("LMDeploy Main Branch Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Get GPU info
    import subprocess
    gpu_info = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"\nGPU: {gpu_info}")

    # Verify models exist
    for model_cfg in MODELS:
        if os.path.exists(model_cfg['path']):
            print(f"  [OK] {model_cfg['name']}: {model_cfg['path']}")
        else:
            print(f"  [MISSING] {model_cfg['name']}: {model_cfg['path']}")

    # Run benchmarks
    all_results = []

    for model_cfg in MODELS:
        if not os.path.exists(model_cfg['path']):
            print(f"\nSkipping {model_cfg['name']} - path not found")
            continue

        # Clear GPU memory before testing new model
        import torch
        torch.cuda.empty_cache()

        print(f"\n{'='*70}")
        print(f"GPU Memory before {model_cfg['name']}:")
        print(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
        print(f"{'='*70}")

        results = run_benchmark(model_cfg, SCENARIOS)
        all_results.extend(results)

    # Save results
    output_file = f'benchmark_main_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    final_results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': gpu_info,
        'lmdeploy_version': 'main',
        'backend': 'turbomind',
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Scenario':<12} {'Speed (t/s)':<15} {'Status':<10}")
    print("-" * 70)
    for r in all_results:
        if r.get('status') == 'success':
            print(f"{r['model']:<30} {r['scenario']:<12} {r['tokens_per_sec']:<15.2f} {r['status']:<10}")
        else:
            print(f"{r['model']:<30} {r.get('scenario', 'N/A'):<12} {'ERROR':<15} {r.get('status', 'error'):<10}")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()