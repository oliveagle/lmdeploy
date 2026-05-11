#!/usr/bin/env python3
"""
LMDeploy + DFlash + TurboQuant 测试脚本
"""
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 必须在 import lmdeploy 之前设置
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.messages import SpeculativeConfig, QuantPolicy

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"

PROMPT = "Hello, how are you?"
MAX_NEW_TOKENS = 128


def run_test(test_name, quant_policy, spec_config=None):
    """运行单个测试"""
    print(f"\n{'=' * 70}")
    print(f"测试: {test_name}")
    print(f"{'=' * 70}")

    try:
        print("加载模型...")
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=PytorchEngineConfig(
                tp=1,
                session_len=2048,
                cache_max_entry_count=0.8,
                max_batch_size=1,
                block_size=64,
                eager_mode=True,
                quant_policy=quant_policy,
                dtype='float16',
            ),
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
    results['basic'] = run_test(
        "基础 LMDeploy",
        QuantPolicy.NONE
    )

    # 测试 2: LMDeploy + TurboQuant
    results['turboquant'] = run_test(
        "LMDeploy + TurboQuant",
        QuantPolicy.TURBO_QUANT
    )

    # 测试 3: LMDeploy + DFlash
    results['dflash'] = run_test(
        "LMDeploy + DFlash",
        QuantPolicy.NONE,
        SpeculativeConfig(
            method="dflash",
            model=DRAFT_MODEL,
            num_speculative_tokens=8,
        )
    )

    # 测试 4: LMDeploy + DFlash + TurboQuant
    results['dflash_turboquant'] = run_test(
        "LMDeploy + DFlash + TurboQuant",
        QuantPolicy.TURBO_QUANT,
        SpeculativeConfig(
            method="dflash",
            model=DRAFT_MODEL,
            num_speculative_tokens=8,
        )
    )

    print(f"\n\n{'=' * 70}")
    print("测试总结")
    print(f"{'=' * 70}")
    for name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
