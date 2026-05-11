#!/usr/bin/env python3
"""
V100 + AWQ MoE 测试 - 使用 Gloo 后端
"""
import os
import sys
import time

# V100 分组
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 强制使用 Gloo 后端, 避免 NCCL P2P 问题
os.environ['USE_DIST_BACKEND'] = 'gloo'

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"


def main():
    print("=" * 70)
    print("V100 + AWQ MoE 测试 (Gloo 后端)")
    print("=" * 70)
    print(f"模型: {MODEL_PATH}")
    print(f"后端: Gloo (避免 NCCL P2P)")
    print()

    from lmdeploy import pipeline, PytorchEngineConfig

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=2048,
        cache_max_entry_count=0.6,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
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
        print(f"✓ 模型加载成功 ({time.time()-start:.2f}s)")
        print()

        prompts = ["你好", "What is AI?"]
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 测试 {i}: {prompt} ---")
            start = time.time()
            response = pipe(prompt, max_new_tokens=32, do_sample=False)
            elapsed = time.time() - start
            print(f"输出: {response.text[:100]}")
            print(f"生成: {response.generate_token_len} tokens, "
                  f"{elapsed:.2f}s ({response.generate_token_len/max(elapsed,0.001):.1f} tok/s)")

        print("\n✓ 全部测试成功!")
        return 0
    except Exception as e:
        import traceback
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
