#!/bin/bash
# 纯净 MoE 层性能对比测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "纯净 MoE 层性能测试"
echo "======================================================================"
echo "脚本目录: $SCRIPT_DIR"
echo "当前目录: $(pwd)"
echo ""

RESULT_DIR="$SCRIPT_DIR/pure_moe_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"
echo "结果目录: $RESULT_DIR"
echo ""

# 测试 1: 基线
echo "=== 测试 1/2: 基线 (无缓存) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    test_pure_moe_no_cache.py \
    --max-seq 524288 \
    --iter 3 \
    --output "$RESULT_DIR/baseline.json" \
    2>&1 | tee "$RESULT_DIR/baseline.log"

if [ ! -f "$RESULT_DIR/baseline.json" ]; then
    echo "❌ 测试 1 失败: 未生成结果文件"
    exit 1
fi
echo "✓ 测试 1 完成"
echo ""

# 测试 2: 权重缓存
echo "=== 测试 2/2: 权重缓存 (16 hot experts) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29502 \
    test_pure_moe_no_cache.py \
    --enable-cache \
    --cache-hot 16 \
    --max-seq 524288 \
    --iter 3 \
    --output "$RESULT_DIR/cache_16.json" \
    2>&1 | tee "$RESULT_DIR/cache_16.log"

if [ ! -f "$RESULT_DIR/cache_16.json" ]; then
    echo "❌ 测试 2 失败: 未生成结果文件"
    exit 1
fi
echo "✓ 测试 2 完成"
echo ""

# 验证结果文件
echo "=== 验证结果文件 ==="
for f in baseline.json cache_16.json; do
    if [ -f "$RESULT_DIR/$f" ]; then
        size=$(stat -c%s "$RESULT_DIR/$f" 2>/dev/null || stat -f%z "$RESULT_DIR/$f")
        echo "  $f: ${size} bytes"
        # 显示 JSON 结构
        python3 -c "
import json
with open('$RESULT_DIR/$f') as fp:
    data = json.load(fp)
    print(f'    配置: {data.get(\"config\", {})}')
    print(f'    测试次数: {data.get(\"test_iterations\", \"?\")}')
    print(f'    测量数: {len(data.get(\"measurements\", []))}')
"
    else
        echo "  ❌ $f: 不存在"
    fi
done
echo ""

# 分析结果
echo "======================================================================"
echo "性能分析"
echo "======================================================================"

python3 << PYEOF
import json
import os
import sys

result_dir = "$RESULT_DIR"

tests = [
    ("baseline.json", "基线 (无缓存)"),
    ("cache_16.json", "权重缓存 (16 专家)"),
]

data_map = {}

for test_file, test_name in tests:
    result_path = os.path.join(result_dir, test_file)
    if not os.path.exists(result_path):
        print(f"⚠️  {test_file}: 不存在")
        continue

    with open(result_path, 'r') as f:
        data = json.load(f)

    data_map[test_name] = data

if not data_map:
    print("❌ 没有可用结果")
    sys.exit(1)

print("\n" + "=" * 90)
print("Tokens/s 性能对比 (越高越好)")
print("=" * 90)
print(f"{'Seq Len':<12} {'基线':<15} {'缓存 16':<15} {'加速比':<12} {'提升':<10}")
print("-" * 90)

all_seq_lens = set()
for data in data_map.values():
    for m in data['measurements']:
        if m['status'] == 'OK':
            all_seq_lens.add(m['seq_len'])

all_seq_lens = sorted(all_seq_lens)

for seq_len in all_seq_lens:
    baseline_tok = None
    cache_tok = None

    for test_name, data in data_map.items():
        for m in data['measurements']:
            if m['seq_len'] == seq_len and m['status'] == 'OK':
                if '基线' in test_name:
                    baseline_tok = m['tokens_per_sec']
                elif '缓存' in test_name:
                    cache_tok = m['tokens_per_sec']

    if baseline_tok is not None and cache_tok is not None:
        speedup = cache_tok / baseline_tok
        improvement = ((cache_tok - baseline_tok) / baseline_tok) * 100
        print(f"{seq_len:<12} {baseline_tok:<15.0f} {cache_tok:<15.0f} {speedup:<12.2f}x {improvement:<10.1f}%")

print("\n" + "=" * 90)
print("延迟对比 (ms，越低越好)")
print("=" * 90)
print(f"{'Seq Len':<12} {'基线':<15} {'缓存 16':<15} {'改善':<10}")
print("-" * 90)

for seq_len in all_seq_lens:
    baseline_lat = None
    cache_lat = None

    for test_name, data in data_map.items():
        for m in data['measurements']:
            if m['seq_len'] == seq_len and m['status'] == 'OK':
                if '基线' in test_name:
                    baseline_lat = m['latency_ms']
                elif '缓存' in test_name:
                    cache_lat = m['latency_ms']

    if baseline_lat is not None and cache_lat is not None:
        improvement = ((baseline_lat - cache_lat) / baseline_lat) * 100
        print(f"{seq_len:<12} {baseline_lat:<15.2f} {cache_lat:<15.2f} {improvement:<10.1f}%")

print()

# 内存使用
print("=" * 90)
print("内存使用")
print("=" * 90)
for test_name, data in data_map.items():
    model_mem = data.get('model_mem_gb', 0)
    print(f"  {test_name}: {model_mem:.2f} GB")

print()
print(f"详细结果保存在: {result_dir}")
print("文件列表:")
for f in os.listdir(result_dir):
    if f.endswith(('.json', '.log')):
        size = os.path.getsize(os.path.join(result_dir, f))
        print(f"  {f}: {size} bytes")

PYEOF

echo ""
echo "======================================================================"
echo "✓ 测试完成"
echo "======================================================================"
