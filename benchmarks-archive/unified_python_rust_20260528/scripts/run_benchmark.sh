#!/usr/bin/env bash
# Unified Benchmark Script for LMDeploy Python vs Rust
# 测试 Python + Rust 统一的性能基准

set -e

# 配置
MODEL_PATH="/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
SCRIPT_PATH="benchmarks-archive/unified_python_rust_20260528/scripts/unified_benchmark.py"
OUTPUT_DIR="benchmarks-archive/unified_python_rust_20260528/results"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "LMDeploy 统一性能基准测试 (Python vs Rust)"
echo "模型: Qwen3.6-35B-A3B-AWQ"
echo "测试输入长度: 512, 1024, 4096, 8192"
echo "测试输出长度: 512"
echo "并发度: 1"
echo "=========================================================="
echo ""

# 运行 Python 基准测试
echo ""
echo "=========================================================="
echo "Running Python TurboMind Benchmark..."
echo "=========================================================="

python3 "$SCRIPT_PATH" \
    --model "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --backend turbomind

echo ""
echo "=========================================================="
echo "所有测试完成！"
echo "=========================================================="
echo "结果文件位于: $OUTPUT_DIR/"
