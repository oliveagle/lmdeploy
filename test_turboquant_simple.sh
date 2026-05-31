#!/bin/bash
# 简单的 TurboQuant 测试脚本 - 验证 KV cache 量化是否正常工作

set -e

echo "========================================"
echo "TurboQuant KV Cache 量化测试"
echo "========================================"

# 模型路径
MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型不存在: $MODEL_PATH"
    exit 1
fi

echo "模型: $MODEL_PATH"
echo "EP: 4, TP: 4"
echo "KV Cache: TurboQuant (K=4bit QJL4, V=2bit MSE)"
echo ""

# 创建测试脚本
cat > /tmp/test_turboquant.py << 'PYEOF'
#!/usr/bin/env python3
"""TurboQuant KV Cache 量化测试"""

import gc
import os
import sys
import time
from pathlib import Path

# 添加 lmdeploy 到路径
LMDEPLOY_PATH = sys.argv[1] if len(sys.argv) > 1 else '.'
sys.path.insert(0, str(Path(LMDEPLOY_PATH)))

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'

def cleanup():
    """清理 GPU 缓存"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def main():
    print("=" * 70)
    print("TurboQuant KV Cache 量化测试")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    model_path = sys.argv[2] if len(sys.argv) > 2 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print(f"模型: {model_path}")
    print(f"EP: 4, TP: 4")
    print(f"KV Cache 量化: TurboQuant (42) - K=4bit QJL4, V=2bit MSE")
    print()

    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存/卡: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print()
    except:
        pass

    # 配置引擎 - 使用 TurboQuant KV cache 量化
    print("配置引擎...")
    engine_config = PytorchEngineConfig(
        tp=1,
        ep=4,
        attn_tp_size=1,
        moe_tp_size=4,
        session_len=2048,
        cache_max_entry_count=0.5,  # 量化后可以设置更大的 cache
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant: K=4bit, V=2bit
    )

    print("quant_policy =", engine_config.quant_policy)
    print("quant_policy value =", int(engine_config.quant_policy))
    assert engine_config.quant_policy == 42, f"quant_policy 应该是 42, 实际是 {engine_config.quant_policy}"
    print("✓ quant_policy 配置正确")
    print()

    print("加载模型...")
    cleanup()

    start = time.time()
    try:
        pipe = pipeline(
            model_path=model_path,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        print(f"✓ 模型加载完成 ({time.time() - start:.1f}秒)")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 检查显存
    try:
        import torch
        if torch.cuda.is_available():
            print("\n各 GPU 显存使用:")
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {mem:.2f} GB / {total:.2f} GB ({mem/total*100:.1f}%)")
    except:
        pass

    # 简单对话测试
    print("\n" + "=" * 70)
    print("对话测试")
    print("=" * 70)

    prompts = [
        "你好，请简短回答。",
        "什么是深度学习？用一句话回答。",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {prompt}")

        try:
            import torch
            if torch.cuda.is_available():
                print(f"推理前显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

            start = time.time()
            response = pipe(prompt, max_new_tokens=32, do_sample=False)
            elapsed = time.time() - start

            output = response.text if hasattr(response, 'text') else str(response)
            print(f"助手: {output.strip()}")
            print(f"⏱️  {elapsed:.2f}秒")

        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        cleanup()

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
PYEOF

# 运行测试
echo "开始测试..."
echo ""

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_turboquant.py "$(pwd)" "$MODEL_PATH"

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
