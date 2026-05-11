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

# Check if models exist
if not os.path.exists(target_model):
    print(f"ERROR: Target model not found at {target_model}")
    sys.exit(1)

if not os.path.exists(draft_model):
    print(f"ERROR: Draft model not found at {draft_model}")
    sys.exit(1)

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
try:
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='INFO',
    )
    print("Pipeline created successfully!")

    # Test inference
    print("\n=== Testing inference ===")
    prompt = "Hello, how are you?"

    # Use GenerationConfig object instead of dict
    gen_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = pipe(prompt, gen_config=gen_config)
    print(f"Prompt: {prompt}")
    print(f"Response: {response.text}")

    print("\n=== Test completed successfully! ===")

except Exception as e:
    print(f"ERROR during test: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
