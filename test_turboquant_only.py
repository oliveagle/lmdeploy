#!/usr/bin/env python3
"""
仅测试基础 LMDeploy + TurboQuant
使用正确的 cache_max_entry_count 配置
"""
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).parent))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.messages import QuantPolicy

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
PROMPT = "Hello, how are you?"
MAX_NEW_TOKENS = 128


def main():
    print("=" * 70)
    print("LMDeploy 基础测试")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print()

    # cache_max_entry_count=0.9 表示 90% 空闲显存用于 KV 缓存
    # 这个值应该大，而不是小！
    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=4096,
        cache_max_entry_count=0.9,  # 90% 空闲显存用于缓存
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant KV 量化
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
        gen_start = time.time()
        response = pipe(
            PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        gen_time = time.time() - gen_start

        print(f"\n输出: {response.text}")
        print(f"生成 tokens: {response.generate_token_len}")
        print(f"推理时间: {gen_time:.2f}s")
        print(f"吞吐量: {response.generate_token_len/gen_time:.2f} tokens/s")
        print("✓ 测试成功")
        return 0
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
