#!/usr/bin/env python3
"""
测试 LMDeploy + TurboQuant + DFlash
分步测试：先测试基础，再测试 DFlash
"""
import os
import sys
import time
from pathlib import Path

# 使用当前源码
sys.path.insert(0, str(Path(__file__).parent))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, TurbomindEngineConfig
from lmdeploy.messages import QuantPolicy

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"
PROMPT = "你好，请介绍一下你自己。"
MAX_NEW_TOKENS = 256


def test_pytorch_turboquant():
    """测试 PyTorch backend + TurboQuant"""
    print("=" * 70)
    print("测试 1: PyTorch Backend + TurboQuant (TP=4, V100 only)")
    print("=" * 70)

    # 只使用 V100 GPU 0-3，避免 P2P 问题
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=8192,
        cache_max_entry_count=0.9,
        max_batch_size=1,
        block_size=64,
        eager_mode=False,  # 使用 graph 模式
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

        print("\n测试推理...")
        gen_start = time.time()
        response = pipe(
            PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
        )
        gen_time = time.time() - gen_start

        print(f"\n输出: {response.text}")
        print(f"生成 tokens: {response.generate_token_len}")
        print(f"推理时间: {gen_time:.2f}s")
        print(f"吞吐量: {response.generate_token_len/gen_time:.2f} tokens/s")
        print("✓ 测试成功")
        return True
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return False


def test_pytorch_dflash():
    """测试 PyTorch backend + TurboQuant + DFlash"""
    print("\n" + "=" * 70)
    print("测试 2: PyTorch Backend + TurboQuant + DFlash (TP=4, V100 only)")
    print("=" * 70)

    # 只使用 V100 GPU 0-3
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # TODO: 等待 DFlash 集成完成
    print("⚠ DFlash 功能尚未完全集成到 PyTorch backend")
    print("  请使用 Turbomind backend 进行 DFlash 测试")
    return False


def test_turbomind_turboquant():
    """测试 Turbomind backend + TurboQuant"""
    print("\n" + "=" * 70)
    print("测试 3: Turbomind Backend + TurboQuant (TP=4, V100 only)")
    print("=" * 70)

    # 只使用 V100 GPU 0-3
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # 注意：Turbomind 需要从源码编译
    print("⚠ Turbomind engine 需要从源码编译")
    print("  当前虚拟环境未编译 Turbomind，跳过此测试")
    print("  编译方法: pip install -e . (从源码安装)")
    return False


def test_turbomind_dflash():
    """测试 Turbomind backend + TurboQuant + DFlash"""
    print("\n" + "=" * 70)
    print("测试 4: Turbomind Backend + TurboQuant + DFlash (TP=4, V100 only)")
    print("=" * 70)

    # TODO: 等待 DFlash 集成完成
    print("⚠ DFlash 功能尚未完全集成到 Turbomind backend")
    print("  当前开发进度见 DFLASH_PROJECT_SUMMARY.md")
    return False


def main():
    results = {}

    # 运行测试
    results['pytorch_turboquant'] = test_pytorch_turboquant()
    results['pytorch_dflash'] = test_pytorch_dflash()
    results['turbomind_turboquant'] = test_turbomind_turboquant()
    results['turbomind_dflash'] = test_turbomind_dflash()

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败/跳过"
        print(f"  {name}: {status}")

    return 0 if any(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
