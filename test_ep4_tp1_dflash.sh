#!/bin/bash
# EP=4 + TP=1 + DFlash 测试 - 无 TurboQuant

set -e

echo "========================================"
echo "EP=4 + TP=1 + DFlash 测试 (无 TurboQuant)"
echo "========================================"

MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL="/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"

use_dflash=false
if [ -d "$DRAFT_MODEL" ]; then
    use_dflash=true
    echo "✓ DFlash 模型存在"
else
    echo "⚠️  DFlash 模型不存在"
fi

cat > /tmp/test_ep4_tp1_dflash.py << 'PYEOF'
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
    from lmdeploy.messages import SpeculativeConfig, QuantPolicy

    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
    draft_model = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print("EP=4 + TP=1 + DFlash 测试")
    print("=" * 70)
    print(f"Target: {model_path}")
    print(f"Draft:  {draft_model or 'None'}")
    print(f"EP=4, TP=1, session_len=256")
    print()

    # Speculative config
    spec_config = None
    if draft_model and os.path.exists(draft_model):
        spec_config = SpeculativeConfig(
            method='dflash',
            model=draft_model,
            num_speculative_tokens=4,
        )
        print("启用 DFlash speculative decoding")
    else:
        print("不启用 speculative decoding")

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
        # quant_policy=QuantPolicy.TURBO_QUANT,  # 先关闭 TurboQuant
    )

    print("加载模型...")
    try:
        pipe = pipeline(
            model_path=model_path,
            trust_remote_code=True,
            backend_config=engine_config,
            speculative_config=spec_config,
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 简单测试
    print("\n测试推理...")
    try:
        response = pipe("你好", max_new_tokens=16, do_sample=False)
        print(f"输出: {response.text}")
        print("✓ 测试成功")
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
PYEOF

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_ep4_tp1_dflash.py "$MODEL_PATH" "${DRAFT_MODEL:-}"
