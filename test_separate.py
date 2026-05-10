#!/usr/bin/env python3
"""
分别测试每个功能，每次只运行一个测试
"""
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).parent))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.messages import SpeculativeConfig, QuantPolicy

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"

PROMPT = "Hello, how are you?"
MAX_NEW_TOKENS = 128


def test_basic():
    print("\n" + "=" * 70)
    print("测试 1: 基础 LMDeploy")
    print("=" * 70)

    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=4096,
        cache_max_entry_count=0.9,  # 90% 空闲显存用于 KV 缓存
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.NONE,
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
        return True
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return False


def test_turboquant():
    print("\n" + "=" * 70)
    print("测试 2: LMDeploy + TurboQuant")
    print("=" * 70)

    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=4096,
        cache_max_entry_count=0.9,
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
        return True
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return False


def test_dflash():
    print("\n" + "=" * 70)
    print("测试 3: LMDeploy + DFlash")
    print("=" * 70)

    spec_config = SpeculativeConfig(
        method="dflash",
        model=DRAFT_MODEL,
        num_speculative_tokens=8,
    )

    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=4096,
        cache_max_entry_count=0.9,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.NONE,
        dtype='float16',
    )

    try:
        print("加载模型...")
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=engine_config,
            speculative_config=spec_config,
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

        if hasattr(response, 'req_metrics') and response.req_metrics:
            spec_info = response.req_metrics.spec_info
            if spec_info:
                print(f"\nSpeculative Decoding 统计:")
                print(f"  Accept rate: {spec_info.get('accept_rate', 'N/A')}")
                print(f"  Speculative steps: {spec_info.get('num_spec_steps', 'N/A')}")

        print("✓ 测试成功")
        return True
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return False


def test_dflash_turboquant():
    print("\n" + "=" * 70)
    print("测试 4: LMDeploy + DFlash + TurboQuant")
    print("=" * 70)

    spec_config = SpeculativeConfig(
        method="dflash",
        model=DRAFT_MODEL,
        num_speculative_tokens=8,
    )

    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=4096,
        cache_max_entry_count=0.9,
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
            speculative_config=spec_config,
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

        if hasattr(response, 'req_metrics') and response.req_metrics:
            spec_info = response.req_metrics.spec_info
            if spec_info:
                print(f"\nSpeculative Decoding 统计:")
                print(f"  Accept rate: {spec_info.get('accept_rate', 'N/A')}")
                print(f"  Speculative steps: {spec_info.get('num_spec_steps', 'N/A')}")

        print("✓ 测试成功")
        return True
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return False


def main():
    print("=" * 70)
    print("LMDeploy + DFlash + TurboQuant 测试")
    print(f"目标模型: {MODEL_PATH}")
    print(f"草稿模型: {DRAFT_MODEL}")
    print("=" * 70)

    results = {}

    # 测试 1: 基础 LMDeploy
    results['basic'] = test_basic()

    # 测试 2: LMDeploy + TurboQuant
    results['turboquant'] = test_turboquant()

    # 测试 3: LMDeploy + DFlash
    results['dflash'] = test_dflash()

    # 测试 4: LMDeploy + DFlash + TurboQuant
    results['dflash_turboquant'] = test_dflash_turboquant()

    print(f"\n\n{'=' * 70}")
    print("测试总结")
    print(f"{'=' * 70}")
    for name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")


if __name__ == '__main__':
    main()
