#!/bin/bash
# 最终解决方案：单 GPU 测试 (不使用分布式)

set -e

echo "========================================"
echo "单 GPU + TurboQuant KV Cache 测试"
echo "========================================"

# 模型路径
MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型不存在: $MODEL_PATH"
    exit 1
fi

echo "模型: $MODEL_PATH"
echo "配置: 单 GPU (0), TurboQuant KV Cache"
echo ""

# 创建测试脚本
cat > /tmp/test_single_gpu_turboquant.py << 'PYEOF'
#!/usr/bin/env python3
"""单 GPU + TurboQuant 测试"""

import gc
import os
import sys
import time
from pathlib import Path

# 添加 lmdeploy 到路径
LMDEPLOY_PATH = sys.argv[1] if len(sys.argv) > 1 else '.'
sys.path.insert(0, str(Path(LMDEPLOY_PATH)))

# 设置环境变量 - 只使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    print("单 GPU + TurboQuant 测试")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    model_path = sys.argv[2] if len(sys.argv) > 2 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print(f"模型: {model_path}")
    print(f"GPU: 0 (无分布式)")
    print(f"KV Cache 量化: TurboQuant (42)")

    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    except:
        pass

    # 配置引擎 - 单 GPU, TurboQuant
    engine_config = PytorchEngineConfig(
        tp=1,
        ep=1,
        session_len=512,  # 更小的 session
        cache_max_entry_count=0.3,
        max_batch_size=1,
        block_size=32,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,
    )

    print("\n加载模型...")
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
            mem = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\nGPU 显存: {mem:.2f} GB / {total:.2f} GB ({mem/total*100:.1f}%)")
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
python3 /tmp/test_single_gpu_turboquant.py "$(pwd)" "$MODEL_PATH"

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
