#!/usr/bin/env python3
"""最简单的 EP4 + TurboQuant 测试 (单 GPU 模拟)"""

import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

try:
    from lmdeploy.pytorch.config import ModelConfig, CacheConfig, BackendConfig, DistConfig, QuantPolicy
    from lmdeploy import pipeline, PytorchEngineConfig

    # 基础配置
    model_path = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print("=== 基础测试 (单 GPU, 无 EP/TP) ===")
    print("Config: TurboQuant KV cache, 极小 cache")

    # 配置 TurboQuant KV cache
    engine_config = PytorchEngineConfig(
        session_len=512,  # 进一步减小 session_len
        cache_max_entry_count=0.02,  # 极小的 KV cache (2%)
        max_batch_size=1,
        block_size=16,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,  # TurboQuant: K=4bit, V=2bit
    )

    print(f"Engine Config: {engine_config}")
    print(f"quant_policy = {engine_config.quant_policy}")

    print("\nInitializing pipeline...")

    pipe = pipeline(
        model_path,
        backend_config=engine_config
    )

    print("Pipeline created successfully!")

    # 简单测试
    prompts = [
        "Hello, please tell me a joke.",
    ]

    print("\nGenerating...")
    outputs = pipe(prompts, request_output_len=16)
    for out in outputs:
        print(f"Output: {out.text}")
    print("\n=== Success! ===")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n=== ERROR ===")
    print(f"Type: {type(e)}")
    print(f"Message: {e}")
