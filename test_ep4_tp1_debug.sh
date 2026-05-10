#!/bin/bash
# EP=4 + TP=1 + DFlash 测试 - 调试版

set -e

echo "========================================"
echo "EP=4 + TP=1 + DFlash 调试测试"
echo "========================================"

MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL="/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"

cat > /tmp/test_ep4_tp1_debug.py << 'PYEOF'
#!/usr/bin/env python3
import os
import sys
import signal
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 启用 CUDA_LAUNCH_BLOCKING 来同步错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

if __name__ == '__main__':
    sys.path.insert(0, str(Path.cwd()))

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import SpeculativeConfig, QuantPolicy

    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
    draft_model = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print("EP=4 + TP=1 + DFlash 调试测试")
    print("=" * 70)
    print(f"Target: {model_path}")
    print(f"Draft:  {draft_model or 'None'}")
    print()

    # 先测试无 DFlash
    print("[测试1] 无 DFlash + TurboQuant")
    engine_config = PytorchEngineConfig(
        tp=1,
        ep=4,
        attn_tp_size=1,
        moe_tp_size=1,
        session_len=128,
        cache_max_entry_count=0.05,
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
        sys.exit(1)

    print("\n[测试1] 推理测试 (max_new_tokens=8)...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"推理前 GPU 0 显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        response = pipe("你好", max_new_tokens=8, do_sample=False)
        print(f"✓ 输出: {response.text}")

        if torch.cuda.is_available():
            print(f"推理后 GPU 0 显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✓ 无 DFlash 测试成功")

    # 测试有 DFlash 但无 TurboQuant
    if draft_model and os.path.exists(draft_model):
        print("\n[测试2] 有 DFlash + 无 TurboQuant")

        spec_config = SpeculativeConfig(
            method='dflash',
            model=draft_model,
            num_speculative_tokens=4,
        )

        engine_config2 = PytorchEngineConfig(
            tp=1,
            ep=4,
            attn_tp_size=1,
            moe_tp_size=1,
            session_len=128,
            cache_max_entry_count=0.05,
            max_batch_size=1,
            block_size=16,
            eager_mode=True,
            # quant_policy=QuantPolicy.TURBO_QUANT,  # 关闭 TurboQuant
        )

        print("加载模型 (带 DFlash)...")
        try:
            import gc
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pipe2 = pipeline(
                model_path=model_path,
                trust_remote_code=True,
                backend_config=engine_config2,
                speculative_config=spec_config,
            )
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\n[测试2] 推理测试 (max_new_tokens=8)...")
        try:
            if torch.cuda.is_available():
                print(f"推理前 GPU 0 显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

            # 设置超时
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("推理超时")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60秒超时

            response = pipe2("你好", max_new_tokens=8, do_sample=False)
            signal.alarm(0)  # 取消超时

            print(f"✓ 输出: {response.text}")

            if torch.cuda.is_available():
                print(f"推理后 GPU 0 显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        except TimeoutError:
            print("✗ 推理超时 - 可能是死锁或kernel挂起")
            sys.exit(1)
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\n✓ DFlash 测试成功")
PYEOF

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_ep4_tp1_debug.py "$MODEL_PATH" "${DRAFT_MODEL:-}"
