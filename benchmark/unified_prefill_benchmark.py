# Unified Prefill Benchmark Script
# Tests both Python TurboMind and Rust server prefill performance

import time
import sys
import os

# Add lmdeploy to path
sys.path.insert(0, '/mnt/data/lmdeploy')

from lmdeploy import pipeline
from lmdeploy.messages import TurbomindEngineConfig

# Configuration
MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

def run_python_benchmark():
    """Run Python TurboMind prefill benchmark"""
    print("=" * 70)
    print("Python TurboMind Prefill Benchmark")
    print("=" * 70)

    engine_config = TurbomindEngineConfig(
        model_format="awq",
        tp=1,
    )

    print("Loading model...")
    start = time.monotonic()
    pipe = pipeline(MODEL_PATH, backend_config=engine_config)
    load_time = time.monotonic() - start
    print(f"Model loaded in {load_time:.2f}s\n")

    # Test configurations: (input_tokens, num_runs)
    test_configs = [
        (512, 3),
        (1024, 3),
        (2048, 3),
        (4096, 3),
        (8192, 3),
    ]

    results = []

    for input_len, num_runs in test_configs:
        print(f"{'='*70}")
        print(f"input_len={input_len}, num_runs={num_runs}")

        # Create synthetic prompt with approximately target token count
        prompt = "The quick brown fox jumps over the lazy dog. " * (input_len // 5 + 1)

        run_times = []
        for run in range(num_runs):
            start = time.monotonic()
            result = pipe([prompt])
            elapsed = time.monotonic() - start
            run_times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.3f}s, prefill={input_len/elapsed:.0f} tok/s")

        avg_time = sum(run_times) / len(run_times)
        avg_prefill = input_len / avg_time
        results.append({
            'input_len': input_len,
            'avg_time': avg_time,
            'prefill_tps': avg_prefill,
        })
        print(f"  Average: {avg_time:.3f}s, prefill={avg_prefill:.0f} tok/s\n")

    print("\n" + "=" * 70)
    print("SUMMARY (Python TurboMind)")
    print("=" * 70)
    print(f"{'Input Len':<12} {'Avg Time (s)':<15} {'Prefill (tok/s)':<15}")
    print("-" * 42)
    for r in results:
        print(f"{r['input_len']:<12} {r['avg_time']:<15.3f} {r['prefill_tps']:<15.0f}")

    return results

if __name__ == "__main__":
    run_python_benchmark()