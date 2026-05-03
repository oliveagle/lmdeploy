#!/bin/bash
# 快速测试脚本 - 验证功能

set -e

echo "======================================================================"
echo "纯净 MoE 层性能测试 - 快速验证"
echo "======================================================================"
echo ""

RESULT_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "结果目录: $RESULT_DIR"
echo ""

# 测试配置
MAX_SEQ=131072  # 128k (避免 OOM)
ITER=3

echo "=== 测试 1/2: 基线 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29500 \
    test_pure_moe_robust.py \
    --max-seq $MAX_SEQ \
    --iter $ITER \
    --output "$RESULT_DIR/baseline.json" \
    --verbose 2>&1 | tee "$RESULT_DIR/baseline.log"

echo ""
echo "=== 验证基线结果 ==="
if [ -f "$RESULT_DIR/baseline.json" ]; then
    python3 << EOF
import json
with open('$RESULT_DIR/baseline.json') as f:
    data = json.load(f)
print(f"测量记录数: {len(data['measurements'])}")
for m in data['measurements'][:3]:  # 显示前 3 个
    print(f"  {m['seq_len']}: {m['tokens_per_sec']:.1f} tokens/s, {m['latency_ms']:.2f} ms")
EOF
else
    echo "❌ 基线测试失败"
    exit 1
fi

echo ""
echo "=== 测试 2/2: 权重缓存 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    test_pure_moe_robust.py \
    --enable-cache \
    --cache-hot 16 \
    --max-seq $MAX_SEQ \
    --iter $ITER \
    --output "$RESULT_DIR/cache_16.json" \
    --verbose 2>&1 | tee "$RESULT_DIR/cache_16.log"

echo ""
echo "=== 验证缓存结果 ==="
if [ -f "$RESULT_DIR/cache_16.json" ]; then
    python3 << EOF
import json
with open('$RESULT_DIR/cache_16.json') as f:
    data = json.load(f)
print(f"测量记录数: {len(data['measurements'])}")
for m in data['measurements'][:3]:  # 显示前 3 个
    print(f"  {m['seq_len']}: {m['tokens_per_sec']:.1f} tokens/s, {m['latency_ms']:.2f} ms")
EOF
else
    echo "❌ 缓存测试失败"
    exit 1
fi

echo ""
echo "======================================================================"
echo "性能对比分析"
echo "======================================================================"

python3 << EOF
import json

result_dir = "$RESULT_DIR"

# 读取结果
with open(f'{result_dir}/baseline.json') as f:
    baseline = json.load(f)

with open(f'{result_dir}/cache_16.json') as f:
    cache = json.load(f)

print()
print("=" * 70)
print("性能对比")
print("=" * 70)
print(f"{'Seq Len':<10} {'基线':<15} {'缓存 16':<15} {'加速比':<10}")
print("-" * 70)

# 获取所有序列长度
all_seqs = set()
for m in baseline['measurements']:
    if m['status'] == 'OK':
        all_seqs.add(m['seq_len'])

for m in cache['measurements']:
    if m['status'] == 'OK':
        all_seqs.add(m['seq_len'])

for seq_len in sorted(all_seqs):
    baseline_tok = None
    cache_tok = None

    # 查找基线结果
    for m in baseline['measurements']:
        if m['seq_len'] == seq_len and m['status'] == 'OK':
            baseline_tok = m['tokens_per_sec']
            break

    # 查找缓存结果
    for m in cache['measurements']:
        if m['seq_len'] == seq_len and m['status'] == 'OK':
            cache_tok = m['tokens_per_sec']
            break

    if baseline_tok and cache_tok:
        speedup = cache_tok / baseline_tok
        print(f"{seq_len:<10} {baseline_tok:<15.1f} {cache_tok:<15.1f} {speedup:<10.2f}x")

print()
print(f"详细结果: {result_dir}")
EOF

echo ""
echo "======================================================================"
echo "✓ 测试完成"
echo "======================================================================"
