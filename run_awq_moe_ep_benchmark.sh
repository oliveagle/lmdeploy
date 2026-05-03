#!/bin/bash
# AWQ MoE EP 性能优化测试配置脚本
# 用于对比不同优化策略的性能

set -e

# 配置
NUM_GPUS=4
MODEL_PATH="Qwen/Qwen3.6-35B-A3B-AWQ"
BASE_DIR="/home/oliveagle/opt/lmdeploy/lmdeploy"

echo "=================================================="
echo "AWQ MoE EP 性能优化测试"
echo "=================================================="
echo "GPU 数量: $NUM_GPUS"
echo "模型路径: $MODEL_PATH"
echo ""

# 创建结果目录
RESULT_DIR="$BASE_DIR/awq_moe_ep_results"
mkdir -p "$RESULT_DIR"

# 测试配置
CONFIGS=(
    "baseline:无优化"
    "cache_hot_32:缓存 32 个热点专家"
    "cache_all:全量缓存 (仅大显存)"
    "nccl:使用 NCCL 通信"
)

# 基线测试 (无优化)
echo "=================================================="
echo "测试 1: 基线 (无优化)"
echo "=================================================="
torchrun --nproc_per_node=$NUM_GPUS \
    "$BASE_DIR/test_awq_moe_ep_optimized.py" \
    --output "$RESULT_DIR/baseline.json" \
    --num-iterations-prefill 3 \
    --num-iterations-decode 20

# 测试 2: 缓存 32 个热点专家
echo ""
echo "=================================================="
echo "测试 2: 缓存 32 个热点专家"
echo "=================================================="
torchrun --nproc_per_node=$NUM_GPUS \
    "$BASE_DIR/test_awq_moe_ep_optimized.py" \
    --output "$RESULT_DIR/cache_hot_32.json" \
    --enable-weight-cache \
    --cache-hot-experts 32 \
    --num-iterations-prefill 3 \
    --num-iterations-decode 20

# 测试 3: NCCL 通信
echo ""
echo "=================================================="
echo "测试 3: 使用 NCCL 通信"
echo "=================================================="
torchrun --nproc_per_node=$NUM_GPUS \
    "$BASE_DIR/test_awq_moe_ep_optimized.py" \
    --output "$RESULT_DIR/nccl.json" \
    --enable-nccl \
    --num-iterations-prefill 3 \
    --num-iterations-decode 20

# 测试 4: 组合优化
echo ""
echo "=================================================="
echo "测试 4: 组合优化 (缓存 + NCCL)"
echo "=================================================="
torchrun --nproc_per_node=$NUM_GPUS \
    "$BASE_DIR/test_awq_moe_ep_optimized.py" \
    --output "$RESULT_DIR/combined.json" \
    --enable-weight-cache \
    --cache-hot-experts 32 \
    --enable-nccl \
    --num-iterations-prefill 3 \
    --num-iterations-decode 20

# 分析结果
echo ""
echo "=================================================="
echo "性能对比分析"
echo "=================================================="

python3 << 'EOF'
import json
import os

result_dir = "/home/oliveagle/opt/lmdeploy/lmdeploy/awq_moe_ep_results"

configs = {
    "baseline": "无优化",
    "cache_hot_32": "缓存 32 个热点专家",
    "nccl": "NCCL 通信",
    "combined": "组合优化",
}

print(f"{'配置':<20} {'4096 Decode (tok/s)':<25} {'8192 Decode (tok/s)':<25} {'相对基线':<15}")
print("-" * 85)

baseline_decode = None

for config_key, config_name in configs.items():
    result_file = os.path.join(result_dir, f"{config_key}.json")
    if not os.path.exists(result_file):
        continue

    with open(result_file, 'r') as f:
        data = json.load(f)

    seq_4096 = data.get('4096', {}).get('decode', {}).get('tok_per_sec', 0)
    seq_8192 = data.get('8192', {}).get('decode', {}).get('tok_per_sec', 0)

    if baseline_decode is None:
        baseline_decode = seq_4096
        speedup = "1.00x"
    else:
        speedup = f"{seq_4096 / baseline_decode:.2f}x"

    print(f"{config_name:<20} {seq_4096:<25.1f} {seq_8192:<25.1f} {speedup:<15}")

print()
print("内存占用:")
for config_key, config_name in configs.items():
    result_file = os.path.join(result_dir, f"{config_key}.json")
    if not os.path.exists(result_file):
        continue

    with open(result_file, 'r') as f:
        data = json.load(f)

    model_mem = data.get('model_mem_gb', 0)
    peak_mem = data.get('peak_mem_gb', 0)
    print(f"  {config_name}: 模型 {model_mem:.2f} GB, 峰值 {peak_mem:.2f} GB")
EOF

echo ""
echo "=================================================="
echo "测试完成！结果保存在: $RESULT_DIR"
echo "=================================================="
