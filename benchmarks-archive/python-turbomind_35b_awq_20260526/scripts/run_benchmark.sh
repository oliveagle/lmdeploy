#!/usr/bin/env bash
# LMDeploy TurboMind 性能测试脚本
# 测试 Prefill 和 Decode 吞吐量，分开统计

set -e

# 配置
MODEL_PATH="/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
SCRIPT_PATH="benchmark/profile_throughput.py"
DATASET_PATH="/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT_DIR="results"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 测试场景列表
test_cases=(
    "512 512"
    "1024 512"
    "2048 512"
    "4096 512"
    "8192 512"
)

echo "=========================================================="
echo "LMDeploy TurboMind 性能测试"
echo "模型: Qwen3.6-35B-A3B-AWQ"
echo "后端: TurboMind (AWQ 4-bit)"
echo "=========================================================="

# 运行测试
for test_case in "${test_cases[@]}"; do
    input_len=$(echo "$test_case" | awk '{print $1}')
    output_len=$(echo "$test_case" | awk '{print $2}')
    timestamp=$(date +%Y%m%d_%H%M%S)
    output_file="$OUTPUT_DIR/profile_throughput_35b_${input_len}_${output_len}_${timestamp}.csv"

    echo ""
    echo "=========================================================="
    echo "Running: input_len=${input_len}, output_len=${output_len}"
    echo "Output: $output_file"
    echo "=========================================================="

    python "$SCRIPT_PATH" \
        "$DATASET_PATH" \
        "$MODEL_PATH" \
        --backend turbomind \
        --concurrency 1 \
        --num-prompts 5 \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --model-format awq \
        --csv "$output_file"
done

echo ""
echo "=========================================================="
echo "所有测试完成！"
echo "=========================================================="
echo "结果文件位于: $OUTPUT_DIR/"