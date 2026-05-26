#!/usr/bin/env python3
"""
Unified Fair Benchmark Framework - CLI Entry Point

Usage:
    python3 run_fair_benchmark.py --help
    python3 run_fair_benchmark.py --python-only
    python3 run_fair_benchmark.py --rust-only --host localhost --port 50051
    python3 run_fair_benchmark.py --compare
    python3 run_fair_benchmark.py --load /path/to/result.json --compare-with /path/to/another.json

Examples:
    # Run Python TurboMind benchmark only
    python3 run_fair_benchmark.py --python-only --model /path/to/model --context 1K 4K 8K

    # Run Rust gRPC benchmark only (server must be running)
    python3 run_fair_benchmark.py --rust-only --host localhost --port 50051

    # Run both and compare
    python3 run_fair_benchmark.py --compare

    # Compare two saved results
    python3 run_fair_benchmark.py --load results/python.json --compare-with results/rust.json
"""

import os
import sys
import time
import asyncio
import argparse
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark_core import (
    BenchmarkConfig,
    EngineType,
    BenchmarkComparator,
    load_result,
    run_benchmarks,
)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Fair Benchmark Framework for Python vs Rust LMDeploy",
        formatter_class=argparse.RawTextHelpFormatter,
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
        "--max-new-tokens",
        type=int,
        default=1,
        help="Max new tokens to generate (default: 1, for pure prefill measurement)",
    )

    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Run Python TurboMind benchmark only",
    )

    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Run Rust gRPC benchmark only",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Rust gRPC server host (default: localhost)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Rust gRPC server port (default: 50051)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both engines and compare",
    )

    parser.add_argument(
        "--load",
        type=str,
        help="Load a previous result file for comparison",
    )

    parser.add_argument(
        "--compare-with",
        type=str,
        help="Compare with another result file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results (default: benchmark_results)",
    )

    parser.add_argument(
        "--cache-max-entry-count",
        type=float,
        default=0.4,
        help="Cache max entry count (default: 0.4)",
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
        max_new_tokens=args.max_new_tokens,
        cache_max_entry_count=args.cache_max_entry_count,
    )

    # Determine which engines to run
    engines = []
    if args.python_only:
        engines = [EngineType.PYTHON_TURBOMIND]
    elif args.rust_only:
        engines = [EngineType.RUST_GRPC]
    elif args.compare:
        engines = [EngineType.PYTHON_TURBOMIND, EngineType.RUST_GRPC]
    elif args.load:
        engines = []
    else:
        # Default: run Python only
        engines = [EngineType.PYTHON_TURBOMIND]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = []
    if engines:
        results = await run_benchmarks(config, engines)

    # Load previous results for comparison
    loaded_results = []
    if args.load:
        r = load_result(Path(args.load))
        if r:
            loaded_results.append(r)
            print(f"Loaded result from {args.load}")
        else:
            print(f"Failed to load result from {args.load}")

    if args.compare_with:
        r = load_result(Path(args.compare_with))
        if r:
            loaded_results.append(r)
            print(f"Loaded comparison result from {args.compare_with}")
        else:
            print(f"Failed to load comparison result from {args.compare_with}")

    all_results = results + loaded_results

    if all_results:
        comparator = BenchmarkComparator(all_results)
        comparator.print_comparison()

        # Save results
        if results:
            for r in results:
                filename = (f"benchmark_{r.engine_type}_{int(r.timestamp)}.json")
                path = output_dir / filename
                r.save(path)
                print(f"\nResult saved to {path}")

        # Save comparison if loading multiple files
        if len(all_results) >= 2:
            comparison_path = output_dir / f"comparison_{int(time.time())}.json"
            comparator.save_comparison(comparison_path)
            print(f"\nComparison saved to {comparison_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
