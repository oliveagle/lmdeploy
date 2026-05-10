#!/bin/bash
# TP=4 + TurboQuant 测试 (无 EP)

set -e

echo "========================================"
echo "TP=4 + TurboQuant 测试 (无 EP)"
echo "========================================"

# 模型路径
MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

echo ""
echo "配置:"
echo "  Target: $MODEL_PATH"
echo "  TP: 4"
echo "  KV Cache 量化: TurboQuant"
echo ""

# 创建测试脚本
cat > /tmp/test_tp4_simple.py << 'PYEOF'
#!/usr/bin/env python3
"""TP=4 简单测试"""

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
    print("TP=4 简单测试")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    model_path = sys.argv[2] if len(sys.argv) > 2 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print(f"配置:")
    print(f"  Target: {model_path}")
    print(f"  TP: 4")
    print(f"  KV Cache 量化: TurboQuant")

    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存/卡: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    except:
        pass

    # 配置引擎: TP=4, TurboQuant
    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=1024,
        cache_max_entry_count=0.3,
        max_batch_size=1,
        block_size=16,
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
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i) / 1024**3
                print(f"  GPU {i} 显存: {mem:.2f} GB")
    except:
        pass

    # 简单测试
    print("\n" + "=" * 70)
    print("简单推理测试")
    print("=" * 70)

    prompt = "你好"
    print(f"\n用户: {prompt}")

    try:
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
python3 /tmp/test_tp4_simple.py "$(pwd)" "$MODEL_PATH"
