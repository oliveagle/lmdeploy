#!/usr/bin/env python3
"""Test DFlash with Qwen3.5-9B-AWQ (target) + Qwen3.5-9B-DFlash (draft)"""

import os
import sys

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
)

print("\n=== Creating pipeline with DFlash ===")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    speculative_config=speculative_config,
    log_level='INFO',
)
print("Pipeline created successfully!")

# Test inference
print("\n=== Testing inference ===")
prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain the concept of machine learning in simple terms.",
]

gen_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

for i, prompt in enumerate(prompts):
    print(f"\n--- Test {i+1} ---")
    print(f"Prompt: {prompt}")
    response = pipe(prompt, gen_config=gen_config)
    print(f"Response: {response.text}")

print("\n=== Test completed successfully! ===")