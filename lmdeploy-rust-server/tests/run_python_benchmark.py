#!/usr/bin/env python3
"""
Quick Python TurboMind Benchmark Runner

Run this script directly to benchmark Python TurboMind prefill performance.

Usage:
    python3 run_python_benchmark.py
    python3 run_python_benchmark.py --context 1K 4K
    python3 run_python_benchmark.py --runs 5
    python3 run_python_benchmark.py --output results/
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark_core import (
    BenchmarkConfig,
    EngineType,
    run_benchmarks,
)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Python TurboMind Benchmark Runner",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ",
        help="Path to the model directory",
    )

    parser.add_argument(
        "--context",
        nargs="+",
        choices=["1K", "2K", "4K", "8K", "16K"],
        default=["1K", "2K", "4K", "8K"],
        help="Context lengths to benchmark (default: 1K 2K 4K 8K)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measurement iterations (default: 3)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results (default: benchmark_results)",
    )

    return parser.parse_args()


CONTEXT_MAP = {
    "1K": ("1K", 1024),
    "2K": ("2K", 2048),
    "4K": ("4K", 4096),
    "8K": ("8K", 8192),
    "16K": ("16K", 16384),
}


async def main() -> int:
    """Main entry point."""
    args = get_args()

    test_cases = [CONTEXT_MAP[c] for c in args.context]

    config = BenchmarkConfig(
        model_path=args.model,
        test_cases=test_cases,
        warmup_iterations=args.warmup,
        measurement_iterations=args.runs,
    )

    print("=" * 80)
    print("Python TurboMind Benchmark")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Test cases: {[tc[0] for tc in config.test_cases]}")
    print(f"Warmup iterations: {config.warmup_iterations}")
    print(f"Measurement iterations: {config.measurement_iterations}")

    results = await run_benchmarks(config, [EngineType.PYTHON_TURBOMIND])

    if results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            filename = f"python_benchmark_{int(r.timestamp)}.json"
            path = output_dir / filename
            r.save(path)
            print(f"\nResult saved to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))