#!/usr/bin/env python3
"""
分别测试 LMDeploy 和 LMDeploy + DFlash
每次只运行一个测试以避免内存不足
"""
import os
import sys
import time
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).parent))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.messages import SpeculativeConfig, QuantPolicy

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"

PROMPT = "Hello, how are you?"
MAX_NEW_TOKENS = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["basic", "turboquant", "dflash", "dflash_turboquant"], default="basic")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"测试: {args.test}")
    print(f"=" * 70)

    quant_policy = QuantPolicy.NONE
    if "turboquant" in args.test:
        quant_policy = QuantPolicy.TURBO_QUANT

    spec_config = None
    if "dflash" in args.test:
        spec_config = SpeculativeConfig(
            method="dflash",
            model=DRAFT_MODEL,
            num_speculative_tokens=8,
        )

    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=512,
        cache_max_entry_count=0.2,
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
        return 0
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"Stacktrace:\n{traceback.format_exc()}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
