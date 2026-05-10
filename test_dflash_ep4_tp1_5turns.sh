#!/bin/bash
# DFlash EP=4 + TP=1 测试 - 5轮对话
#
# 使用 EP=4 减少显存占用:
# - EP=4: 256个专家分到4个GPU，每卡64个专家
# - TP=1 (moe_tp): FFN维度不分片
# - EP = 4x 显存缩减（MoE权重部分）
# - TurboQuant KV cache: K=4bit QJL4, V=2bit MSE

set -e

echo "========================================"
echo "DFlash EP=4 + TP=1 测试 - 5轮对话"
echo "========================================"

# 模型路径
MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL="/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型不存在: $MODEL_PATH"
    exit 1
fi

use_dflash=false
if [ -d "$DRAFT_MODEL" ]; then
    use_dflash=true
    echo "✓ DFlash 模型存在: $DRAFT_MODEL"
else
    echo "⚠️  DFlash 模型不存在: $DRAFT_MODEL"
    echo "使用普通模式（无 DFlash）"
fi

echo ""
echo "配置:"
echo "  Target: $MODEL_PATH"
echo "  Draft:  ${DRAFT_MODEL:-无}"
echo "  EP: 4, TP: 1 (moe_tp)"
echo "  GPUs: 0,1,2,3"
echo ""

# 创建测试脚本
cat > /tmp/test_dflash_5turns.py << 'PYEOF'
#!/usr/bin/env python3
"""DFlash EP=4 + TP=1 测试 - 5轮对话"""

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
    print("DFlash EP=4 + TP=1 测试")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import SpeculativeConfig, QuantPolicy

    model_path = sys.argv[2] if len(sys.argv) > 2 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
    draft_model = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"配置:")
    print(f"  Target: {model_path}")
    if draft_model and os.path.exists(draft_model):
        print(f"  Draft:  {draft_model}")
    print(f"  EP: 4, TP: 1 (moe_tp)")
    print(f"  KV Cache 量化: TurboQuant (42) - K=4bit QJL4, V=2bit MSE")

    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存/卡: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            print(f"  当前显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except:
        pass

    # DFlash speculative decoding 配置
    spec_config = None
    use_dflash = draft_model is not None and os.path.exists(draft_model)
    if use_dflash:
        spec_config = SpeculativeConfig(
            method='dflash',
            model=draft_model,
            num_speculative_tokens=4,
        )
        print("  启用 DFlash speculative decoding")
    else:
        print("  不启用 speculative decoding")

    # 关键配置策略:
    #
    # EP=4, moe_tp=1 组合:
    # - EP=4: 256个专家分到4个GPU，每卡64个专家 (4x 显存缩减)
    # - moe_tp=1: FFN维度不分片
    # - EP = 4x 总缩减
    # - MoE权重从 ~34GB/卡 → ~12GB/卡
    #
    # attn_tp=1: 注意力部分不分片（EP模式默认）
    #
    # TurboQuant KV cache:
    # - K: 4-bit QJL4 量化
    # - V: 2-bit MSE 量化
    # - KV cache 显存减少约 75%
    #
    # cache_max_entry_count=0.3: 保守的KV cache分配
    # - 约 1.5GB GPU 显存给 KV cache
    engine_config = PytorchEngineConfig(
        tp=1,  # 基础 tp=1
        ep=4,  # EP=4 专家并行
        attn_tp_size=1,  # 注意力不分片
        moe_tp_size=1,  # MoE FFN 维度不分片
        session_len=256,  # 减少session长度避免OOM
        cache_max_entry_count=0.1,  # 减小KV cache避免OOM
        max_batch_size=1,
        block_size=16,
        eager_mode=True,  # 使用eager模式避免CUDA graph问题
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant KV cache量化
    )

    print("\n加载模型...")
    cleanup()

    start = time.time()
    try:
        pipe = pipeline(
            model_path=model_path,
            trust_remote_code=True,
            backend_config=engine_config,
            speculative_config=spec_config,
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

    # 5轮对话
    conversations = [
        "你好",
        "介绍一下人工智能",
        "什么是深度学习",
        "谢谢你的解释",
        "再见",
    ]

    print("\n" + "=" * 70)
    print("开始对话测试")
    print("=" * 70)
    print(f"总共 {len(conversations)} 轮对话")

    for i, prompt in enumerate(conversations, 1):
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
    print("✓ 所有对话测试完成")
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
python3 /tmp/test_dflash_5turns.py "$(pwd)" "$MODEL_PATH" "${DRAFT_MODEL:-}"

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
