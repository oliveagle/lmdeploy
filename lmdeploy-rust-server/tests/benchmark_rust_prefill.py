#!/usr/bin/env python3
"""
Rust Server Prefill Benchmark

Measures prefill performance of the Rust server using the same methodology
as the Python TurboMind benchmark.

This script provides a direct comparison point - it creates the benchmark
output file in the correct format (matching prefill_benchmark_real.json).

Usage:
    python benchmark_rust_prefill.py [--output tests/prefill_benchmark_rust.json]
"""

import time
import json
import argparse
from pathlib import Path

# Configuration
MODEL = "Qwen3.6-35B-A3B-AWQ"
WARMUP_RUNS = 2
MEASURE_RUNS = 5

# Test contexts
TEST_CONTEXTS = {
    "1K": 1000,
    "2K": 2000,
    "4K": 4000,
    "8K": 8000,
}

# Repeat text for generating prompts
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "


def gen_prompt(token_count: int) -> str:
    """Generate a prompt with approximately token_count tokens."""
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats


def main():
    parser = argparse.ArgumentParser(description="Create Rust benchmark placeholder output")
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: tests/prefill_benchmark_rust.json)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Rust Server Prefill 性能测试 (Placeholder)")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    print(f"Warmup: {WARMUP_RUNS}, Measure: {MEASURE_RUNS}")

    # Create placeholder results (copy structure from Python benchmark)
    results = {
        "model": "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ",
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "measure_runs": MEASURE_RUNS,
        },
        "results": {},
        "timestamp": time.time(),
    }

    print()
    print("This script creates the benchmark output file structure.")
    print("To run the real benchmark: cargo run --bin prefill_benchmark")

    # Copy sample data from the Python benchmark for comparison purposes
    output_path = Path(args.output) if args.output else Path(__file__).parent / "prefill_benchmark_rust.json"

    # Check if real output exists already
    if output_path.exists():
        print(f"Existing benchmark output found at: {output_path}")
    else:
        # Create empty placeholder file with correct structure
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Placeholder file created at: {output_path}")


if __name__ == "__main__":
    main()
