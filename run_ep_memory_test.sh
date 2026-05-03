#!/bin/bash
# EP=4 内存优化测试脚本

set -e

echo "======================================================================"
echo "EP=4 内存优化性能测试"
echo "======================================================================"
echo ""
echo "测试配置:"
echo "  - 最大序列长度: 8192 (避免 OOM)"
echo "  - 权重缓存: 16 个专家"
echo "  - 使用 NCCL backend"
echo ""

RESULT_DIR="awq_moe_ep_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# 测试 1: 基线 (无权重缓存)
echo "测试 1: 基线 (NCCL, 无权重缓存)"
echo "----------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test_ep_memory_optimized.py \
    --name "baseline" \
    --cache-hot 0 \
    --max-seq-len 8192 \
    --output "$RESULT_DIR/baseline.json" \
    2>&1 | tee "$RESULT_DIR/baseline.log"

# 测试 2: 权重缓存 16 个专家
echo ""
echo "测试 2: 权重缓存 (16 hot experts)"
echo "----------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test_ep_memory_optimized.py \
    --enable-cache \
    --cache-hot 16 \
    --name "cache_16" \
    --max-seq-len 8192 \
    --output "$RESULT_DIR/cache_16.json" \
    2>&1 | tee "$RESULT_DIR/cache_16.log"

# 汇总结果
echo ""
echo "======================================================================"
echo "性能对比"
echo "======================================================================"

python3 << EOF
import json
import os

result_dir = "$RESULT_DIR"

tests = [
    ("baseline", "基线 (无缓存)"),
    ("cache_16", "权重缓存 (16 专家)"),
]

print(f"{'配置':<20} {'4096 Decode':<15} {'加速比':<10} {'峰值内存':<15}")
print("-" * 60)

baseline_decode = None

for test_file, test_name in tests:
    result_path = os.path.join(result_dir, f"{test_file}.json")
    if not os.path.exists(result_path):
        continue

    with open(result_path, 'r') as f:
        data = json.load(f)

    if '4096' in data['results'] and 'decode_tok_per_sec' in data['results']['4096']:
        decode_tok = data['results']['4096']['decode_tok_per_sec']
        peak_mem = data['peak_mem_gb']

        if baseline_decode is None:
            baseline_decode = decode_tok
            speedup = 1.0
        else:
            speedup = decode_tok / baseline_decode

        print(f"{test_name:<20} {decode_tok:<15.1f} {speedup:<10.2f}x {peak_mem:<15.2f} GB")

print()
print(f"结果目录: {result_dir}")
EOF

echo ""
echo "======================================================================"
echo "✓ 测试完成"
echo "======================================================================"
