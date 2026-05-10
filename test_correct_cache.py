#!/usr/bin/env python3
"""
LMDeploy 测试 - 使用合理的 cache_max_entry_count
cache_max_entry_count 控制空闲显存中多少比例用于 KV 缓存
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


def run_test(name, use_turboquant=False, use_dflash=False):
    print(f"\n{'=' * 70}")
    print(f"测试: {name}")
    print(f"{'=' * 70}")

    quant_policy = QuantPolicy.TURBO_QUANT if use_turboquant else QuantPolicy.NONE

    spec_config = None
    if use_dflash:
        spec_config = SpeculativeConfig(
            method="dflash",
            model=DRAFT_MODEL,
            num_speculative_tokens=8,
        )

    # 关键：cache_max_entry_count=0.9 表示 90% 空闲显存用于缓存
    # 如果模型占用 20GB，空闲 12GB，那么 0.9 * 12GB = 10.8GB 用于缓存
    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=4096,  # 支持更长上下文
        cache_max_entry_count=0.9,  # 90% 空闲显存用于 KV 缓存
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=quant_policy,
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
        print(f"Stacktrace:\n{traceback.format_exc()[:1000]}")
        return False


def main():
    print("=" * 70)
    print("LMDeploy + DFlash + TurboQuant 测试")
    print(f"目标模型: {MODEL_PATH}")
    print(f"草稿模型: {DRAFT_MODEL}")
    print("=" * 70)

    results = {}

    # 测试 1: 基础 LMDeploy
    results['basic'] = run_test("基础 LMDeploy", use_turboquant=False, use_dflash=False)

    # 测试 2: LMDeploy + TurboQuant
    results['turboquant'] = run_test("LMDeploy + TurboQuant", use_turboquant=True, use_dflash=False)

    # 测试 3: LMDeploy + DFlash
    results['dflash'] = run_test("LMDeploy + DFlash", use_turboquant=False, use_dflash=True)

    # 测试 4: LMDeploy + DFlash + TurboQuant
    results['dflash_turboquant'] = run_test(
        "LMDeploy + DFlash + TurboQuant", use_turboquant=True, use_dflash=True
    )

    print(f"\n\n{'=' * 70}")
    print("测试总结")
    print(f"{'=' * 70}")
    for name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")


if __name__ == '__main__':
    main()
