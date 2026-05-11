#!/usr/bin/env python3
"""Test DFlash with Qwen3.5-9B-AWQ (target) + Qwen3.5-9B-DFlash (draft)"""

import os
import sys

# Set LD_LIBRARY_PATH for the build
os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

# Model paths
target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print(f"Target model: {target_model}")
print(f"Draft model: {draft_model}")

# Configure speculative decoding with DFlash
speculative_config = SpeculativeConfig(
    method='dflash',
    model=draft_model,
    num_speculative_tokens=8,
    quant_policy=0,  # FP16 (no quantization)
)

# Create TurboMind engine config
tm_config = TurbomindEngineConfig(
    model_format='awq',
    tensor_parallel=1,
    cache_max_entry_count=0.5,  # 使用更小的 KV cache，0.5 表示空闲显存的 50%
    quant_policy=8,  # 启用 KV cache 8bit 量化，减少内存占用
)

print("\n=== Creating pipeline with DFlash ===")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    speculative_config=speculative_config,
    log_level='WARNING',
)
print("Pipeline created successfully!")

# Test inference
print("\n=== Testing inference ===")
prompt = "什么是人工智能？请用一句话解释。"

gen_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=False,  # Greedy for better DFlash acceptance
    temperature=0.7,
    top_p=0.9,
)

print(f"Prompt: {prompt}")
messages = [{"role": "user", "content": prompt}]
response = pipe(
    messages,
    gen_config=gen_config,
    sequence_start=True,  # 标记为独立请求的开始
    sequence_end=True,    # 标记为独立请求的结束
    chat_template_kwargs={'enable_thinking': False}
)
print(f"Response: {response.text}")

print("\n=== Test completed successfully! ===")
