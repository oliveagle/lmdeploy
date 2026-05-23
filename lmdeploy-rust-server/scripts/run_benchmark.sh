#!/bin/bash
# Benchmark runner for LMDeploy Rust Server
# Usage: ./scripts/run_benchmark.sh [model_path]
#
# Supports:
#   - AWQ quantized models (e.g., Qwen3.6-35B-A3B-AWQ)
#   - Non-quantized models (e.g., Qwen3.5-9B)
#   - Context lengths: 1K, 4K, 8K, 16K, 32K tokens
#
# Examples:
#   ./scripts/run_benchmark.sh
#   ./scripts/run_benchmark.sh /path/to/awq/model
#   ./scripts/run_benchmark.sh /path/to/non-awq/model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_DIR/benches"

# Default model path (AWQ model)
MODEL_PATH="${1:-/mnt/eaget-4tb/modelscope_models/tclf00/Qwen3___6-35B-A3B-AWQ}"

echo "=== LMDeploy Rust Server Benchmark ==="
echo "Model: $MODEL_PATH"
echo "Context lengths: 1K, 4K, 8K, 16K, 32K tokens"
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