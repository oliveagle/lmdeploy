#!/usr/bin/env python3
"""
测试 LMDeploy 不使用 TurboQuant (仅使用 V100 GPU)
"""
import os
import sys
import time
from pathlib import Path

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.messages import QuantPolicy

# 关键：使用物理 GPU 1-4 (都是 V100)
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 禁用 FlashAttention-3，使用基础实现
os.environ['LMDEPLOY_USE_FLASHATTENTION3'] = '0'
os.environ['LMDEPLOY_USE_FLASHATTENTION'] = '0'

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"


def main():
    print("=" * 70)
    print("LMDeploy 测试 (TP=4, V100 only, NO TurboQuant)")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"GPU: CUDA_VISIBLE_DEVICES=1,2,3,4 (4x V100)")
    print(f"TurboQuant: DISABLED")
    print(f"FlashAttention-3: DISABLED")
    print()

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.5,  # 降低 KV cache 使用
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.NONE,  # 不使用 TurboQuant
        dtype='float16',
    )

    try:
        print("加载模型...")
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        load_time = time.time() - start
        print(f"✓ 模型加载成功 ({load_time:.2f}s)")

        print("\n测试推理...")
        response = pipe("你好，请介绍一下你自己。", max_new_tokens=64, do_sample=False)
        print(f"输出: {response.text}")
        print("\n✓ 测试成功！")
        return 0
    except Exception as e:
        import traceback
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
