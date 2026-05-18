#!/bin/bash
# Benchmark runner for LMDeploy Rust Server
# Usage: ./scripts/run_benchmark.sh [model_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_DIR/benches"

# Default model path
MODEL_PATH="${1:-/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ}"

echo "=== LMDeploy Rust Server Benchmark ==="
echo "Model: $MODEL_PATH"
echo "======================================"

# Set model path for benchmark
export LMDEPLOY_MODEL_PATH="$MODEL_PATH"

cd "$PROJECT_DIR"

# Run criterion benchmarks
echo "Running criterion benchmarks..."
cargo bench --bench benchmark -- --noplot 2>&1 | tee "$PROJECT_DIR/benchmark_results.txt"

# Generate HTML report
echo ""
echo "Generating HTML report..."
cargo bench --bench benchmark -- --noplot --output-format html

echo ""
echo "=== Results saved to: benchmark_results.txt ==="
echo "HTML report: target/criterion/"