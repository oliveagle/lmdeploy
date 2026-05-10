#!/usr/bin/env python3
"""
LMDeploy Turbomind + DFlash 测试脚本
使用 V100 GPU 组 (GPU 0-3), 排除 PG199

测试流程:
1. Turbomind + AWQ (TP=4 on V100)
2. Turbomind + DFlash + AWQ (TP=4 on V100)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# V100 分组 (GPU 0-3), 不包含 PG199
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"


def print_gpu_info():
    """打印 GPU 信息"""
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    print("GPU 状态 (仅 V100):")
    for i, line in enumerate(result.stdout.strip().split('\n')[:4]):  # 只显示前 4 块
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4:
            print(f"  GPU {parts[0]}: {parts[1]} - {parts[2]} MB 已用, {parts[3]} MB 空闲")
    print()


def test_turbomind_awq():
    """测试 1: Turbomind + AWQ"""
    print("=" * 70)
    print("测试 1: Turbomind + AWQ (TP=4, V100 only)")
    print("=" * 70)

    from lmdeploy import pipeline, TurbomindEngineConfig

    # Turbomind 配置
    engine_config = TurbomindEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.9,
        max_batch_size=1,
        model_format='awq',
        dtype='float16',
    )

    print("加载模型 (Turbomind + AWQ)...")
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
        "Python 中如何实现快速排序?",
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


def test_turbomind_dflash():
    """测试 2: Turbomind + DFlash + AWQ"""
    print("\n" + "=" * 70)
    print("测试 2: Turbomind + DFlash + AWQ (TP=4, V100 only)")
    print("=" * 70)

    from lmdeploy import pipeline, TurbomindEngineConfig
    from lmdeploy.messages import SpeculativeConfig

    # Turbomind 配置
    engine_config = TurbomindEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.8,
        max_batch_size=1,
        model_format='awq',
        dtype='float16',
    )

    # DFlash speculative decoding 配置
    speculative_config = SpeculativeConfig(
        method='dflash',
        model=DRAFT_MODEL,
        num_speculative_tokens=4,
    )

    print("加载主模型 + DFlash draft 模型 (Turbomind)...")
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
        "How does attention mechanism work?",
        "如何优化大语言模型的推理速度?",
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

        # 如果有 DFlash 统计信息，打印出来
        if hasattr(response, 'speculative_stats'):
            stats = response.speculative_stats
            print(f"  DFlash 统计: 接受率={stats.get('accept_rate', 0):.2%}")

    print("\n✓ 测试 2 成功!")
    return True


def main():
    print("=" * 70)
    print("LMDeploy Turbomind + DFlash + AWQ 测试")
    print("=" * 70)
    print(f"主模型: {MODEL_PATH}")
    print(f"Draft 模型: {DRAFT_MODEL}")
    print(f"引擎: Turbomind (C++ backend)")
    print(f"GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 (4x V100 NV2)")
    print()

    print_gpu_info()

    results = {}
    try:
        results['turbomind_awq'] = test_turbomind_awq()
    except Exception as e:
        import traceback
        print(f"✗ 测试 1 失败: {e}")
        traceback.print_exc()
        results['turbomind_awq'] = False

    try:
        results['turbomind_dflash'] = test_turbomind_dflash()
    except Exception as e:
        import traceback
        print(f"✗ 测试 2 失败: {e}")
        traceback.print_exc()
        results['turbomind_dflash'] = False

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {'✓' if v else '✗'}")
    return 0 if any(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
