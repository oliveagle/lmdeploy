#!/bin/bash
# EP=4 + TP=1 测试 - 无 DFlash

set -e

echo "========================================"
echo "EP=4 + TP=1 测试 (无 DFlash)"
echo "========================================"

MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

cat > /tmp/test_ep4_tp1_nodflash.py << 'PYEOF'
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == '__main__':
    sys.path.insert(0, str(Path.cwd()))

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print("=" * 70)
    print("EP=4 + TP=1 测试 (无 DFlash)")
    print("=" * 70)
    print(f"Target: {model_path}")
    print(f"EP=4, TP=1, session_len=256")
    print()

    engine_config = PytorchEngineConfig(
        tp=1,
        ep=4,
        attn_tp_size=1,
        moe_tp_size=1,
        session_len=256,
        cache_max_entry_count=0.1,
        max_batch_size=1,
        block_size=16,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,
    )

    print("加载模型...")
    try:
        pipe = pipeline(
            model_path=model_path,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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

    for i, prompt in enumerate(conversations, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {prompt}")

        try:
            response = pipe(prompt, max_new_tokens=32, do_sample=False)
            output = response.text if hasattr(response, 'text') else str(response)
            print(f"助手: {output.strip()}")
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n✓ 所有测试完成")
PYEOF

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_ep4_tp1_nodflash.py "$MODEL_PATH"
