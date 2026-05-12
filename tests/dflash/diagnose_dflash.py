#!/usr/bin/env python3
"""
DFlash Integration Diagnostic Script

This script checks if DFlash is properly integrated and identifies issues.
"""

import os
import sys
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'DEBUG'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash Integration Diagnostic")
print("=" * 60)

# Check CUDA
print(f"\n1. CUDA Status:")
print(f"   Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Check models
print(f"\n2. Model Paths:")
print(f"   Target: {target_model}")
print(f"   Exists: {os.path.exists(target_model)}")
print(f"   Draft: {draft_model}")
print(f"   Exists: {os.path.exists(draft_model)}")

# Create DFlash pipeline
print(f"\n3. Creating DFlash Pipeline...")
speculative_config = SpeculativeConfig(
    method='dflash',
    model=draft_model,
    num_speculative_tokens=8,
    quant_policy=0,
)

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tensor_parallel=1,
    cache_max_entry_count=0.5,
    quant_policy=8,
    session_len=8192,
)

try:
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='INFO'
    )
    print("   ✓ Pipeline created successfully")
except Exception as e:
    print(f"   ✗ Pipeline creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if DFlash is enabled
print(f"\n4. Checking DFlash Status...")
print(f"   Checking pipe.engine.model_comm...")

# Try to access DFlash internals
try:
    model_comm = pipe.engine.model_comm
    print(f"   model_comm type: {type(model_comm)}")

    # Check if load_dflash_weights and enable_dflash methods exist
    has_load = hasattr(model_comm, 'load_dflash_weights')
    has_enable = hasattr(model_comm, 'enable_dflash')
    print(f"   has load_dflash_weights: {has_load}")
    print(f"   has enable_dflash: {has_enable}")

except Exception as e:
    print(f"   Error accessing model_comm: {e}")

# Run a simple inference test
print(f"\n5. Running Inference Test...")
try:
    gen_config = GenerationConfig(max_new_tokens=16, do_sample=False)
    resp = pipe(
        [{"role": "user", "content": "Hello"}],
        gen_config=gen_config,
        sequence_start=True,
        sequence_end=True,
        chat_template_kwargs={'enable_thinking': False}
    )
    print(f"   ✓ Inference successful")
    print(f"   Output: {resp.text}")
    print(f"   Input tokens: {resp.input_token_len}")
    print(f"   Output tokens: {len(resp.token_ids) - resp.input_token_len}")
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("Diagnostic Complete")
print(f"{'='*60}")

# Cleanup
del pipe
