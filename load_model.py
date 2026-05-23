#!/usr/bin/env python3
"""TurboMind model loader - Python bridge for Rust server.

Loads HuggingFace models via the Python TurboMind API and exposes
inference capabilities through stdin/stdout JSON protocol.

Usage:
    echo '{"action": "load", "model_path": "/path/to/model", "session_len": 8192}' | python3 load_model.py
"""

import sys
import json
import os
import os.path as osp

# Add lmdeploy to path
sys.path.insert(0, '/mnt/data/lmdeploy')

import torch

def load_and_test(model_path: str, session_len: int = 8192):
    """Load model via Python TurboMind API and run inference."""
    from lmdeploy.turbomind.turbomind import TurboMind
    from lmdeploy.messages import TurbomindEngineConfig
    from lmdeploy.tokenizer import Tokenizer

    print(f"Loading model: {model_path}")
    print(f"Session length: {session_len}")

    config = TurbomindEngineConfig(
        session_len=session_len,
        max_batch_size=128,
        cache_max_entry_count=0.8,
    )

    print("Creating TurboMind instance...")
    tm = TurboMind(model_path, engine_config=config, trust_remote_code=False)
    print(f"Model loaded: {tm.model_name}")
    print(f"GPU count: {tm.gpu_count}")
    print(f"Vocab size: {tm._vocab_size}")

    # Test tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer(model_path)

    # Test inference
    prompt = "Hello, tell me about Python in one sentence."
    print(f"\nTest prompt: {prompt}")

    inputs = tokenizer.encode(prompt)
    print(f"Input tokens: {len(inputs)}")

    # Generate
    for output in tm.generate(
        input_ids=[inputs],
        gen_config=tm.create_gen_config(max_new_tokens=50, temperature=0.1)
    ):
        for gen in output:
            if gen.status == 0:  # success
                text = tokenizer.decode(gen.token_ids.tolist(), skip_special_tokens=True)
                print(f"\nGenerated: {text}")
                print(f"Total tokens: {len(gen.token_ids)}")

    tm.close()
    print("\nModel closed successfully.")

if __name__ == '__main__':
    model_path = '/mnt/data/models/modelscope_models/Qwen3.6-35B-A3B-AWQ'
    load_and_test(model_path)
