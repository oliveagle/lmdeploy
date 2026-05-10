#!/bin/bash
# 文件名: test_turboquant_ultra_low_mem.sh
#
# 极低内存版本: EP=4 + TurboQuant + 最小配置
# 针对 16GB V100 的极限测试

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

cat > /tmp/test_turboquant_ultra_low_mem.py << 'EOF'
"""极低内存配置测试 - EP=4 + TurboQuant"""

import gc
import os
import sys
import time
from pathlib import Path

# 添加 lmdeploy 到路径
LMDEPLOY_PATH = sys.argv[1] if len(sys.argv) > 1 else '.'
sys.path.insert(0, str(LMDEPLOY_PATH))

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    print("极低内存配置测试 - EP=4 + TurboQuant")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    model_path = sys.argv[2] if len(sys.argv) > 2 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print(f"配置:")
    print(f"  Target: {model_path}")
    print(f"  EP: 4 (256专家分到4个GPU)")
    print(f"  TP: 1 (不使用TP)")
    print(f"  KV Cache: TurboQuant (K=4bit, V=2bit)")
    print(f"  Session: 512 tokens")
    print(f"  Cache: 1% (极小)")

    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存/卡: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
    except:
        pass

    # 极低内存配置
    engine_config = PytorchEngineConfig(
        tp=1,  # 不使用TP
        ep=4,  # EP4专家并行
        attn_tp_size=1,
        moe_tp_size=1,
        session_len=512,  # 更小的session
        cache_max_entry_count=0.01,  # 极小的KV cache (1%)
        max_batch_size=1,
        block_size=16,  # 最小允许值
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant: K=4bit, V=2bit
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
            print("\n显存使用:")
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {mem:.2f} GB / {total:.2f} GB ({mem/total*100:.1f}%)")
    except:
        pass

    # 简单测试
    prompts = [
        "Hello, how are you?",
    ]

    print("\n" + "=" * 70)
    print("开始推理测试")
    print("=" * 70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---")
        print(f"Input: {prompt}")

        try:
            import torch
            if torch.cuda.is_available():
                print(f"推理前显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

            start = time.time()
            response = pipe(prompt, max_new_tokens=16, do_sample=False)
            elapsed = time.time() - start

            output = response.text if hasattr(response, 'text') else str(response)
            print(f"Output: {output.strip()}")
            print(f"⏱️  {elapsed:.2f}秒")

            if torch.cuda.is_available():
                print(f"推理后显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

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
EOF

echo "开始极低内存测试..."
echo ""

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_turboquant_ultra_low_mem.py "$(pwd)" "$MODEL_PATH"

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
