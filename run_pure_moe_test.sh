#!/bin/bash
# 纯净 MoE 层性能对比测试 - 无 KV Cache 干扰

set -e

echo "======================================================================"
echo "纯净 MoE 层性能测试 (无 KV Cache，无前缀缓存干扰)"
echo "======================================================================"
echo ""
echo "测试说明:"
echo "  1. 直接测试 MoE 层，无 KV Cache"
echo "  2. 每次使用不同的随机输入 (避免前缀缓存命中)"
echo "  3. 强制 CUDA 同步确保准确计时"
echo "  4. 测试序列长度: 4k - 512k"
echo ""

RESULT_DIR="pure_moe_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# 测试 1: 基线 (无权重缓存)
echo "测试 1: 基线 (无权重缓存)"
echo "----------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    test_pure_moe_no_cache.py \
    --max-seq 524288 \
    --iter 5 \
    --output "$RESULT_DIR/baseline.json" \
    2>&1 | tee "$RESULT_DIR/baseline.log"

# 测试 2: 权重缓存 16 个专家
echo ""
echo "测试 2: 权重缓存 (16 hot experts)"
echo "----------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    test_pure_moe_no_cache.py \
    --enable-cache \
    --cache-hot 16 \
    --max-seq 524288 \
    --iter 5 \
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
print("Tokens/s 性能对比 (越高越好)")
print("=" * 80)
print(f"{'Seq Len':<12} {'基线':<15} {'缓存 16':<15} {'加速比':<10} {'提升':<10}")
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
                    baseline_tok = m['tokens_per_sec']
                else:
                    cache_tok = m['tokens_per_sec']

    if baseline_tok is not None and cache_tok is not None:
        speedup = cache_tok / baseline_tok
        improvement = ((cache_tok - baseline_tok) / baseline_tok) * 100
        print(f"{seq_len:<12} {baseline_tok:<15.0f} {cache_tok:<15.0f} {speedup:<10.2f}x {improvement:<10.1f}%")

print("\n" + "=" * 80)
print("延迟对比 (ms，越低越好)")
print("=" * 80)
print(f"{'Seq Len':<12} {'基线':<15} {'缓存 16':<15} {'改善':<10}")
print("-" * 80)

for seq_len in [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
    baseline_lat = None
    cache_lat = None

    for test_file, _ in tests:
        result_path = os.path.join(result_dir, f"{test_file}.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path, 'r') as f:
            data = json.load(f)

        for m in data['measurements']:
            if m['seq_len'] == seq_len and m['status'] == 'OK':
                if test_file == 'baseline':
                    baseline_lat = m['latency_ms']
                else:
                    cache_lat = m['latency_ms']

    if baseline_lat is not None and cache_lat is not None:
        improvement = ((baseline_lat - cache_lat) / baseline_lat) * 100
        print(f"{seq_len:<12} {baseline_lat:<15.2f} {cache_lat:<15.2f} {improvement:<10.1f}%")

print()
print("详细结果保存在:", result_dir)
EOF

echo ""
echo "======================================================================"
echo "✓ 测试完成"
echo "结果目录: $RESULT_DIR"
echo "======================================================================"
