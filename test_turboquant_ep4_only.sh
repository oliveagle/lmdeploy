#!/bin/bash
# 文件名: test_turboquant_ep4_only.sh
#
# EP=4 + TurboQuant KV cache 测试 (不含 DFlash)
# 仅测试基础功能，显存优化版本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

cat > /tmp/test_turboquant_ep4.py << 'EOF'
"""EP=4 + TurboQuant KV cache 基础测试"""

import gc
import os
import sys
import time
from pathlib import Path

LMDEPLOY_PATH = sys.argv[1] if len(sys.argv) > 1 else '.'
sys.path.insert(0, str(LMDEPLOY_PATH))

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def main():
    print("=" * 70)
    print("EP=4 + TurboQuant KV cache 基础测试")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    model_path = sys.argv[2] if len(sys.argv) > 2 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print(f"Target: {model_path}")
    print(f"Config: EP=4, TP=1, TurboQuant KV cache")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
    except:
        pass

    engine_config = PytorchEngineConfig(
        tp=1,
        ep=4,
        attn_tp_size=1,
        moe_tp_size=1,
        session_len=512,
        cache_max_entry_count=0.02,
        max_batch_size=1,
        block_size=16,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,
    )

    print("\nLoading model...")
    cleanup()

    start = time.time()
    try:
        pipe = pipeline(
            model_path=model_path,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        print(f"✓ Loaded in {time.time() - start:.1f}s")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        import torch
        if torch.cuda.is_available():
            print("\nGPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {mem:.2f} GB / {total:.2f} GB")
    except:
        pass

    prompts = ["Hello, please tell me a joke."]
    print("\n" + "=" * 70)
    print("Running inference test")
    print("=" * 70)

    for prompt in prompts:
        print(f"\nInput: {prompt}")
        try:
            start = time.time()
            response = pipe(prompt, max_new_tokens=16, do_sample=False)
            elapsed = time.time() - start

            output = response.text if hasattr(response, 'text') else str(response)
            print(f"Output: {output.strip()}")
            print(f"Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 70)
    print("✓ Test completed")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
EOF

echo "开始 EP4 + TurboQuant 测试..."
echo ""

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_turboquant_ep4.py "$(pwd)" "$MODEL_PATH"
