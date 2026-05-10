#!/bin/bash
# EP=4, TP=1, KV TurboQuant 基线测试 (修复版)
# 确保 GPU 配置正确

set -e

echo "========================================"
echo "EP=4, TP=1, KV TurboQuant 基线测试"
echo "========================================"

# 模型路径
MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型不存在: $MODEL_PATH"
    exit 1
fi

# 检查 GPU
echo "🎮 GPU 配置检查:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -4
echo ""

# 创建测试脚本
cat > /tmp/test_baseline_ep4_turboquant.py << 'PYEOF'
#!/usr/bin/env python3
"""EP=4, TP=1, KV TurboQuant 基线测试"""

import gc
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# 添加 lmdeploy 到路径
LMDEPLOY_PATH = sys.argv[1] if len(sys.argv) > 1 else '.'
sys.path.insert(0, str(Path(LMDEPLOY_PATH).resolve()))

# 关键：必须在导入 torch 之前设置 CUDA_VISIBLE_DEVICES
# 使用 GPU 1-4 (4x V100 16GB)，避开 GPU 0 (DRIVE-PG199-PROD 32GB)
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

def cleanup():
    """清理 GPU 缓存"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def get_gpu_memory():
    """获取各 GPU 显存使用"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        memory = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory[i] = {
                'allocated': allocated,
                'total': total,
                'percent': (allocated / total * 100) if total > 0 else 0
            }
        return memory
    except:
        return {}

def print_gpu_memory(label=""):
    """打印显存使用"""
    memory = get_gpu_memory()
    if not memory:
        return
    print(f"\n{label}显存使用:")
    for gpu_id, mem in memory.items():
        print(f"  GPU {gpu_id}: {mem['allocated']:.2f} GB / {mem['total']:.2f} GB ({mem['percent']:.1f}%)")

def main():
    print("=" * 70)
    print("EP=4, TP=1, KV TurboQuant 基线测试")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    model_path = sys.argv[2] if len(sys.argv) > 2 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print(f"\n📍 模型: {model_path}")
    print(f"⚙️  EP=4, TP=1, moe_tp=1, attn_tp=1")
    print(f"🎯 KV Cache: TurboQuant (42) - K=4bit QJL4, V=2bit MSE")

    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🎮 GPU 信息:")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
    except Exception as e:
        print(f"⚠️  无法获取 GPU 信息: {e}")

    # 配置引擎 - EP=4, TP=1, TurboQuant
    engine_config = PytorchEngineConfig(
        tp=1,
        ep=4,
        attn_tp_size=1,
        moe_tp_size=1,
        session_len=1024,
        cache_max_entry_count=0.8,
        max_batch_size=1,
        block_size=64,
        eager_mode=False,
        quant_policy=QuantPolicy.TURBO_QUANT,
        enable_prefix_caching=False,
    )

    print(f"\n🔧 引擎配置:")
    print(f"   tp={engine_config.tp}, ep={engine_config.ep}")
    print(f"   attn_tp_size={engine_config.attn_tp_size}")
    print(f"   moe_tp_size={engine_config.moe_tp_size}")
    print(f"   session_len={engine_config.session_len}")
    print(f"   quant_policy={engine_config.quant_policy}")

    print("\n⏳ 加载模型...")
    cleanup()

    load_start = time.time()
    try:
        pipe = pipeline(
            model_path=model_path,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        load_time = time.time() - load_start
        print(f"✅ 模型加载完成 ({load_time:.1f}秒)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print_gpu_memory("模型加载后 ")

    # 简单对话测试
    print("\n" + "=" * 70)
    print("📝 对话测试 (3 轮)")
    print("=" * 70)

    prompts = [
        "你好，请简短回答。",
        "什么是深度学习？用一句话回答。",
        "请介绍一下 Python。",
    ]

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─' * 70}")
        print(f"🔸 第 {i} 轮")
        print(f"👤 用户: {prompt}")

        cleanup()

        try:
            infer_start = time.time()
            response = pipe(
                prompt,
                max_new_tokens=64,
                do_sample=False,
            )
            infer_time = time.time() - infer_start

            output = response.text if hasattr(response, 'text') else str(response)
            print(f"\n🤖 助手: {output.strip()}")
            print(f"⏱️  推理时间: {infer_time:.2f}秒")

            results.append({
                'round': i,
                'time': infer_time,
            })

        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        print_gpu_memory(f"推理后 ")
        cleanup()

    # 打印总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)

    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results)

    print(f"\n总轮数: {len(results)}")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均时间: {avg_time:.2f}秒")
    print(f"\n各轮时间:")
    for r in results:
        print(f"  轮次 {r['round']}: {r['time']:.2f}秒")

    print_gpu_memory("最终 ")

    print("\n" + "=" * 70)
    print("✅ 基线测试完成")
    print("=" * 70)
    print(f"\n配置: EP=4, TP=1, KV=TurboQuant")
    print(f"此基线用于对比 DFlash 性能提升")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
PYEOF

# 运行测试
echo "🚀 开始测试..."
echo ""

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_baseline_ep4_turboquant.py "$(pwd)" "$MODEL_PATH"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试成功完成"
else
    echo "❌ 测试失败"
fi
echo "========================================"

exit $EXIT_CODE
