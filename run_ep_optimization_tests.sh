#!/bin/bash
# AWQ MoE EP=4 性能优化测试脚本
# 对比不同优化策略的性能提升

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
NUM_GPUS=4
BASELINE_DECODE=46.26  # 从 results_4gpus.json 获取的基线

echo "======================================================================"
echo "AWQ MoE EP=4 性能优化测试"
echo "======================================================================"
echo "GPU 数量: $NUM_GPUS"
echo "基线 Decode: $BASELINE_DECODE tok/s"
echo "======================================================================"
echo ""

# 创建结果目录
RESULT_DIR="awq_moe_ep_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# 测试 1: 基线 (无优化，但使用 NCCL)
echo "测试 1: 基线 (NCCL only)"
echo "----------------------------------------------------------------------"
torchrun --nproc_per_node=$NUM_GPUS test_ep_optimized.py \
    --name "baseline_nccl" \
    --output "$RESULT_DIR/baseline_nccl.json" \
    2>&1 | tee "$RESULT_DIR/baseline_nccl.log"

# 测试 2: 启用权重缓存 (32 个热点专家)
echo ""
echo "测试 2: 权重缓存 (32 hot experts)"
echo "----------------------------------------------------------------------"
torchrun --nproc_per_node=$NUM_GPUS test_ep_optimized.py \
    --enable-cache \
    --cache-hot 32 \
    --name "cache_32" \
    --output "$RESULT_DIR/cache_32.json" \
    2>&1 | tee "$RESULT_DIR/cache_32.log"

# 测试 3: 启用权重缓存 (64 个热点专家)
echo ""
echo "测试 3: 权重缓存 (64 hot experts)"
echo "----------------------------------------------------------------------"
torchrun --nproc_per_node=$NUM_GPUS test_ep_optimized.py \
    --enable-cache \
    --cache-hot 64 \
    --name "cache_64" \
    --output "$RESULT_DIR/cache_64.json" \
    2>&1 | tee "$RESULT_DIR/cache_64.log"

# 汇总结果
echo ""
echo "======================================================================"
echo "性能对比汇总"
echo "======================================================================"

python3 << EOF
import json
import os

result_dir = "$RESULT_DIR"
baseline = $BASELINE_DECODE

tests = [
    ("baseline_nccl", "基线 (NCCL only)"),
    ("cache_32", "权重缓存 (32 hot)"),
    ("cache_64", "权重缓存 (64 hot)"),
]

print(f"{'测试配置':<25} {'4096 Decode (tok/s)':<20} {'相对基线':<15} {'内存占用':<15}")
print("-" * 75)

for test_file, test_name in tests:
    result_path = os.path.join(result_dir, f"{test_file}.json")
    if not os.path.exists(result_path):
        continue

    with open(result_path, 'r') as f:
        data = json.load(f)

    decode_tok = data['results']['4096']['decode_tok_per_sec']
    speedup = data['speedup']
    peak_mem = data['peak_mem_gb']

    print(f"{test_name:<25} {decode_tok:<20.1f} {speedup:<15.2f}x {peak_mem:<15.2f} GB")

print()
print("详细结果保存在:", result_dir)
EOF

echo ""
echo "======================================================================"
echo "测试完成！"
echo "结果目录: $RESULT_DIR"
echo "======================================================================"
