#!/usr/bin/env python3
"""
测试 LMDeploy + TurboQuant (仅使用 V100 GPU)
"""
import os
import sys
import time
from pathlib import Path

# 使用当前源码
sys.path.insert(0, str(Path(__file__).parent))

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.messages import QuantPolicy

# 关键：使用物理 GPU 1-4 (都是 V100)
# nvidia-smi 显示 GPU 0-3 是 V100，但 PyTorch 检测到 GPU 0 是 DRIVE-PG199
# 所以使用 CUDA_VISIBLE_DEVICES=1,2,3,4 来选择 4 个 V100
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 禁用 FlashAttention-3，使用基础实现
os.environ['LMDEPLOY_USE_FLASHATTENTION3'] = '0'
os.environ['LMDEPLOY_USE_FLASHATTENTION'] = '0'

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"


def main():
    print("=" * 70)
    print("LMDeploy + TurboQuant 测试 (TP=4, V100 only)")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"GPU: CUDA_VISIBLE_DEVICES=1,2,3,4 (4x V100)")
    print(f"FlashAttention-3: DISABLED")
    print()

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.8,  # 80% 空闲显存用于 KV 缓存
        max_batch_size=1,
        block_size=64,
        eager_mode=True,  # 禁用 CUDA graph 以避免内存问题
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant KV 量化
        dtype='float16',
    )

    try:
        print("加载模型...")
        print("  配置: TP=4, TurboQuant KV 量化, cache_max_entry_count=0.8")
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        load_time = time.time() - start
        print(f"✓ 模型加载成功 ({load_time:.2f}s)")

        # 检查显存使用
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True)
        print("\nGPU 显存使用:")
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, used, free = line.split(', ')
                print(f"  GPU {idx}: {used} MiB 已用, {free} MiB 空闲")

        print("\n" + "-" * 70)
        print("测试推理")
        print("-" * 70)

        prompts = [
            "你好，请介绍一下你自己。",
            "What is the capital of France?",
            "请用 Python 写一个冒泡排序。",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\n测试 {i}: {prompt[:50]}...")
            gen_start = time.time()
            response = pipe(
                prompt,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
            )
            gen_time = time.time() - gen_start

            print(f"输出: {response.text[:200]}")
            print(f"生成 tokens: {response.generate_token_len}")
            print(f"推理时间: {gen_time:.2f}s")
            print(f"吞吐量: {response.generate_token_len/gen_time:.2f} tokens/s")

        print("\n" + "=" * 70)
        print("✓ 所有测试成功！")
        print("=" * 70)
        return 0

    except Exception as e:
        import traceback
        print(f"\n✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
