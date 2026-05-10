#!/usr/bin/env python3
"""
LMDeploy + TurboQuant + DFlash 测试脚本
- 先测试基础 LMDeploy + TurboQuant (TP=4 on V100)
- 再测试 DFlash 加速
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"


def get_gpu_info():
    """获取 GPU 信息"""
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 5:
            gpus.append({
                'index': parts[0],
                'name': parts[1],
                'total': parts[2],
                'used': parts[3],
                'free': parts[4]
            })
    return gpus


def print_gpu_info():
    """打印 GPU 信息"""
    gpus = get_gpu_info()
    print("GPU 状态:")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']} - "
              f"{gpu['used']}/{gpu['total']} MB ({gpu['free']} MB free)")
    print()


def test_basic_turboquant():
    """测试 1: LMDeploy + TurboQuant (无 DFlash)"""
    print("=" * 70)
    print("测试 1: LMDeploy + TurboQuant KV 量化")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.9,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant KV 量化
        dtype='float16',
    )

    print("加载模型...")
    start = time.time()
    pipe = pipeline(
        model_path=MODEL_PATH,
        trust_remote_code=True,
        backend_config=engine_config,
    )
    load_time = time.time() - start
    print(f"✓ 模型加载成功 ({load_time:.2f}s)")

    print_gpu_info()

    # 测试推理
    prompts = [
        "你好，请简单介绍一下你自己。",
        "What is the capital of France?",
        "用 Python 写一个快速排序。",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n--- 测试 {i+1} ---")
        print(f"输入: {prompt}")
        start = time.time()
        response = pipe(
            prompt,
            max_new_tokens=64,
            do_sample=False,
        )
        elapsed = time.time() - start
        print(f"输出: {response.text[:200]}")
        print(f"生成: {response.generate_token_len} tokens, {elapsed:.2f}s, "
              f"{response.generate_token_len/max(elapsed, 0.001):.1f} tok/s")

    print("\n✓ 测试 1 成功!")
    return True


def test_dflash():
    """测试 2: LMDeploy + DFlash + TurboQuant"""
    print("\n" + "=" * 70)
    print("测试 2: LMDeploy + DFlash + TurboQuant")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy, SpeculativeConfig

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.8,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant KV 量化
        dtype='float16',
    )

    # DFlash speculative decoding 配置
    speculative_config = SpeculativeConfig(
        method='dflash',
        model=DRAFT_MODEL,
        num_speculative_tokens=4,
    )

    print("加载主模型 + DFlash draft 模型...")
    start = time.time()
    pipe = pipeline(
        model_path=MODEL_PATH,
        trust_remote_code=True,
        backend_config=engine_config,
        speculative_config=speculative_config,
    )
    load_time = time.time() - start
    print(f"✓ 模型加载成功 ({load_time:.2f}s)")

    print_gpu_info()

    # 测试推理
    prompts = [
        "请解释 Transformer 架构。",
        "How does attention work in LLMs?",
        "如何优化大模型的推理速度?",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n--- 测试 {i+1} ---")
        print(f"输入: {prompt}")
        start = time.time()
        response = pipe(
            prompt,
            max_new_tokens=64,
            do_sample=False,
        )
        elapsed = time.time() - start
        print(f"输出: {response.text[:200]}")
        print(f"生成: {response.generate_token_len} tokens, {elapsed:.2f}s")

    print("\n✓ 测试 2 成功!")
    return True


def main():
    print("=" * 70)
    print("LMDeploy + DFlash + TurboQuant 测试")
    print("=" * 70)
    print(f"主模型: {MODEL_PATH}")
    print(f"Draft 模型: {DRAFT_MODEL}")
    print(f"GPU: CUDA_VISIBLE_DEVICES=1,2,3,4 (4x V100 NV2)")
    print()

    print_gpu_info()

    results = {}
    try:
        results['turboquant'] = test_basic_turboquant()
    except Exception as e:
        import traceback
        print(f"✗ 测试 1 失败: {e}")
        traceback.print_exc()
        results['turboquant'] = False

    try:
        results['dflash'] = test_dflash()
    except Exception as e:
        import traceback
        print(f"✗ 测试 2 失败: {e}")
        traceback.print_exc()
        results['dflash'] = False

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {'✓' if v else '✗'}")
    return 0 if any(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
