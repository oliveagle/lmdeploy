#!/usr/bin/env python3
"""
LMDeploy PyTorch Backend + AWQ MoE 简单测试
验证 CUDA 非法内存访问问题是否修复
"""
import os
import sys
import time
from pathlib import Path

# 使用前 4 块 V100 GPU (避免 P2P 问题)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"


def print_gpu_info():
    """打印 GPU 信息"""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        print("GPU 状态:")
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                print(f"  GPU {parts[0]}: {parts[1]} - "
                      f"{parts[2]} MB 已用, {parts[3]} MB 空闲")
        print()
    except Exception:
        pass


def test_pytorch_awq():
    """测试 PyTorch Backend + AWQ MoE"""
    print("=" * 70)
    print("LMDeploy PyTorch Backend + AWQ MoE 测试")
    print("=" * 70)
    print(f"模型: {MODEL_PATH}")
    print()

    print_gpu_info()

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    # 使用 4 卡 V100 测试
    engine_config = PytorchEngineConfig(
        tp=4,  # 使用 4 块 V100
        session_len=2048,
        cache_max_entry_count=0.8,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,  # 禁用 CUDA graph
        quant_policy=QuantPolicy.TURBO_QUANT,
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
        print()

        print_gpu_info()

        # 简单推理测试
        prompts = [
            "你好",
            "What is AI?",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 测试 {i}: {prompt} ---")
            start = time.time()
            response = pipe(
                prompt,
                max_new_tokens=32,
                do_sample=False,
            )
            elapsed = time.time() - start

            print(f"输出: {response.text[:100]}")
            print(f"生成: {response.generate_token_len} tokens, {elapsed:.2f}s, "
                  f"{response.generate_token_len/max(elapsed, 0.001):.1f} tok/s")

        print("\n✓ 测试成功!")
        return True

    except Exception as e:
        import traceback
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("AWQ MoE 修复验证测试")
    print("=" * 70)
    print("目的: 验证 topk_ids 验证修复是否解决了 CUDA 非法内存访问问题")
    print()

    success = test_pytorch_awq()

    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"{'✓ 成功' if success else '✗ 失败'}")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
