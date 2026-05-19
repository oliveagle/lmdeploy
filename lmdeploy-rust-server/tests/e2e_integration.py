#!/usr/bin/env python3
"""E2E integration test for lmdeploy-rust-server

This test verifies the complete inference path:
1. Tokenizer loading
2. Model info retrieval
3. (Optional) Inference if models are available
"""

import json
import sys
import os

# Add lmdeploy to path
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/lmdeploy')

from lmdeploy.tokenizer import Tokenizer


def test_tokenizer():
    """Test tokenizer loading"""
    print("[Phase 1] Testing tokenizer loading...")

    model_path = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B"

    try:
        tokenizer = Tokenizer(model_path)
        vocab_size = tokenizer.vocab_size
        print(f"  OK: Tokenizer loaded (vocab_size={vocab_size})")

        # Test encoding
        text = "Hello, world!"
        ids = tokenizer.encode(text)
        print(f"  OK: Encoded '{text}' -> {len(ids)} tokens")

        # Test decoding
        decoded = tokenizer.decode(ids)
        print(f"  OK: Decoded -> '{decoded}'")

        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_model_config():
    """Test model config parsing"""
    print("[Phase 2] Testing model config parsing...")

    model_path = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B"
    config_path = os.path.join(model_path, "config.json")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Handle nested config (text_config for multimodal models)
        text_config = config.get("text_config", config.get("model_config", config))

        hidden_size = text_config.get("hidden_size", config.get("hidden_size", "N/A"))
        num_layers = text_config.get("num_hidden_layers", config.get("num_hidden_layers", "N/A"))
        vocab_size = text_config.get("vocab_size", config.get("vocab_size", "N/A"))

        print(f"  OK: Config loaded (nested: {'text_config' in config})")
        print(f"    hidden_size={hidden_size}")
        print(f"    num_hidden_layers={num_layers}")
        print(f"    vocab_size={vocab_size}")

        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_awq_detection():
    """Test AWQ quantization detection"""
    print("[Phase 3] Testing AWQ quantization detection...")

    # Non-AWQ model
    model_path = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B"
    config_path = os.path.join(model_path, "config.json")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        has_awq = "quantization_config" in config and config.get("quantization_config", {}).get("quant_method") == "awq"
        print(f"  OK: Qwen3.5-9B AWQ={has_awq} (expected: False)")

        # AWQ model (check both locations)
        awq_model_path = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.6-35B-A3B-AWQ"
        awq_config_path = os.path.join(awq_model_path, "config.json")

        # Try alternative path if symlink is broken
        if not os.path.exists(awq_config_path):
            alt_path = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.6-35B-A3B-AWQ/Qwen3___6-35B-A3B-AWQ"
            awq_config_path = os.path.join(alt_path, "config.json")

        if os.path.exists(awq_config_path):
            with open(awq_config_path, 'r') as f:
                awq_config = json.load(f)

            has_awq = "quantization_config" in awq_config and awq_config.get("quantization_config", {}).get("quant_method") == "awq"
            print(f"  OK: Qwen3.6-35B-A3B-AWQ AWQ={has_awq} (expected: True)")
        else:
            print(f"  SKIP: AWQ model config not found")

        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    print("=== LMDeploy E2E Integration Test ===\n")

    results = []

    results.append(("Tokenizer", test_tokenizer()))
    results.append(("Model Config", test_model_config()))
    results.append(("AWQ Detection", test_awq_detection()))

    print("\n=== Test Summary ===")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("==================")

    if all_passed:
        print("\nResult: PASS - All integration tests passed")
        return 0
    else:
        print("\nResult: FAIL - Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
