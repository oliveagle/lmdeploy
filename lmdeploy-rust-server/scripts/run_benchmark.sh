#!/bin/bash
# LMDeploy Rust Server Benchmark Runner
# Run performance benchmarks with various configurations
#
# Usage:
#   ./scripts/run_benchmark.sh [OPTIONS]
#
# Options:
#   --model PATH          Model path (default: AWQ model)
#   --context-lengths N   Space-separated context lengths (default: 1K 4K 8K 16K 32K 48K 64K)
#   --output-length N     Output token length (default: 512)
#   --iterations N        Number of iterations (default: 5)
#   --warmup N            Warmup iterations (default: 2)
#   --format FORMAT       Output format: json, table, both (default: json)
#   --awq-mode            Enable AWQ-optimized benchmark
#   --extended-context    Include 48K, 64K, 128K context tests
#   --quick               Quick test (only 1K, 4K, 8K)
#   --concurrent N        Concurrent requests (default: 1)
#   --output FILE         Output file path
#   --verbose             Enable debug logging
#   --help                Show help

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ"
CONTEXT_LENGTHS="1024 4096 8192 16384 32768 49152 65536"
OUTPUT_LENGTH=512
ITERATIONS=5
WARMUP=2
FORMAT="json"
AWQ_MODE=false
EXTENDED_CONTEXT=false
QUICK_MODE=false
CONCURRENT=1
OUTPUT_FILE=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --context-lengths) CONTEXT_LENGTHS="$2"; shift 2 ;;
        --output-length) OUTPUT_LENGTH="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --format) FORMAT="$2"; shift 2 ;;
        --awq-mode) AWQ_MODE=true; shift ;;
        --extended-context) EXTENDED_CONTEXT=true; shift ;;
        --quick) QUICK_MODE=true; shift ;;
        --concurrent) CONCURRENT="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --verbose) VERBOSE="-v"; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH          Model path"
            echo "  --context-lengths N   Space-separated context lengths"
            echo "  --output-length N     Output token length"
            echo "  --iterations N        Number of iterations"
            echo "  --warmup N            Warmup iterations"
            echo "  --format FORMAT       Output format: json, table, both"
            echo "  --awq-mode            Enable AWQ-optimized benchmark"
            echo "  --extended-context    Include 48K, 64K, 128K context tests"
            echo "  --quick               Quick test (only 1K, 4K, 8K)"
            echo "  --concurrent N        Concurrent requests"
            echo "  --output FILE         Output file path"
            echo "  --verbose             Enable debug logging"
            echo ""
            echo "Presets:"
            echo "  --quick               Quick: 1K, 4K, 8K, 1 iteration"
            echo "  --awq-mode            AWQ: 1K, 4K, 8K, 16K, 32K, 64K"
            echo "  --extended-context    Full: 1K..128K"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Apply preset overrides
if [ "$QUICK_MODE" = true ]; then
    CONTEXT_LENGTHS="1024 4096 8192"
    ITERATIONS=1
fi

if [ "$AWQ_MODE" = true ]; then
    CONTEXT_LENGTHS="1024 4096 8192 16384 32768 65536"
fi

if [ "$EXTENDED_CONTEXT" = true ]; then
    CONTEXT_LENGTHS="1024 4096 8192 16384 32768 49152 65536 131072"
fi

echo "=== LMDeploy Rust Benchmark ==="
echo "Model: $MODEL_PATH"
echo "Context lengths: $CONTEXT_LENGTHS"
echo "Output length: $OUTPUT_LENGTH"
echo "Iterations: $ITERATIONS"
echo "Warmup: $WARMUP"
echo "Format: $FORMAT"
echo "Concurrent: $CONCURRENT"
echo "==============================="

# Build and run
cd "$PROJECT_DIR" || exit 1

BENCH_CMD=(cargo run --release --example benchmark -- "$MODEL_PATH"
    --context-lengths $CONTEXT_LENGTHS
    --output-length "$OUTPUT_LENGTH"
    --iterations "$ITERATIONS"
    --warmup "$WARMUP"
    --format "$FORMAT"
    --concurrent "$CONCURRENT"
)

# Add optional flags
if [ "$AWQ_MODE" = true ]; then
    BENCH_CMD+=(--awq-mode)
fi

if [ "$EXTENDED_CONTEXT" = true ]; then
    BENCH_CMD+=(--extended-context)
fi

if [ "$QUICK_MODE" = true ]; then
    BENCH_CMD+=(--quick)
fi

if [ -n "$OUTPUT_FILE" ]; then
    BENCH_CMD+=(--output "$OUTPUT_FILE")
fi

if [ -n "$VERBOSE" ]; then
    BENCH_CMD+=("$VERBOSE")
fi

# Execute
"${BENCH_CMD[@]}"