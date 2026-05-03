#!/bin/bash
# 大上下文性能测试脚本

set -e

echo "======================================================================"
echo "AWQ MoE EP=4 大上下文性能测试"
echo "======================================================================"
echo ""
echo "测试配置:"
echo "  - 序列长度范围: 4k - 512k"
echo "  - 测试内容: prefill 速度, decode 速度"
echo "  - GPU: 4x V100 16GB"
echo ""

RESULT_DIR="large_context_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# 测试 1: 无权重缓存
echo "测试 1: 基线 (无权重缓存)"
echo "----------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    test_large_context.py \
    --min-seq 4096 \
    --max-seq 524288 \
    --prefill-iter 3 \
    --decode-iter 10 \
    --output "$RESULT_DIR/baseline.json" \
    2>&1 | tee "$RESULT_DIR/baseline.log"

# 测试 2: 权重缓存 16 个专家
echo ""
echo "测试 2: 权重缓存 (16 hot experts)"
echo "----------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    test_large_context.py \
    --enable-cache \
    --cache-hot 16 \
    --min-seq 4096 \
    --max-seq 524288 \
    --prefill-iter 3 \
    --decode-iter 10 \
    --output "$RESULT_DIR/cache_16.json" \
    2>&1 | tee "$RESULT_DIR/cache_16.log"

# 分析结果
echo ""
echo "======================================================================"
echo "性能分析"
echo "======================================================================"

python3 << 'EOF'
import json
import os

result_dir = "$RESULT_DIR"

tests = [
    ("baseline", "基线 (无缓存)"),
    ("cache_16", "权重缓存 (16 专家)"),
]

print("\n" + "=" * 80)
print("Decode 性能对比 (tokens/s)")
print("=" * 80)
print(f"{'Seq Len':<10} {'基线':<15} {'缓存 16':<15} {'加速比':<10}")
print("-" * 80)

for seq_len in [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
    baseline_tok = None
    cache_tok = None

    for test_file, _ in tests:
        result_path = os.path.join(result_dir, f"{test_file}.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path, 'r') as f:
            data = json.load(f)

        for m in data['measurements']:
            if m['seq_len'] == seq_len and m['status'] == 'OK':
                if test_file == 'baseline':
                    baseline_tok = m['decode']['tok_per_sec']
                else:
                    cache_tok = m['decode']['tok_per_sec']

    if baseline_tok is not None and cache_tok is not None:
        speedup = cache_tok / baseline_tok
        print(f"{seq_len:<10} {baseline_tok:<15.1f} {cache_tok:<15.1f} {speedup:<10.2f}x")

print("\n" + "=" * 80)
print("Prefill 性能对比 (tokens/s)")
print("=" * 80)
print(f"{'Seq Len':<10} {'基线':<15} {'缓存 16':<15} {'加速比':<10}")
print("-" * 80)

for seq_len in [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
    baseline_tok = None
    cache_tok = None

    for test_file, _ in tests:
        result_path = os.path.join(result_dir, f"{test_file}.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path, 'r') as f:
            data = json.load(f)

        for m in data['measurements']:
            if m['seq_len'] == seq_len and m['status'] == 'OK':
                if test_file == 'baseline':
                    baseline_tok = m['prefill']['tok_per_sec']
                else:
                    cache_tok = m['prefill']['tok_per_sec']

    if baseline_tok is not None and cache_tok is not None:
        speedup = cache_tok / baseline_tok
        print(f"{seq_len:<10} {baseline_tok:<15.0f} {cache_tok:<15.0f} {speedup:<10.2f}x")

print()
print("详细结果保存在:", result_dir)
EOF

echo ""
echo "======================================================================"
echo "测试完成！"
echo "结果目录: $RESULT_DIR"
echo "======================================================================"
