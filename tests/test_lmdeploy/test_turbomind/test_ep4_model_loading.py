#!/usr/bin/env python3
"""EP=4 Model Loading and Output Quality Test.

This script tests:
1. Model loading with EP=4 configuration
2. Output quality verification (no garbled text)
3. Basic inference functionality

Prerequisites:
- Qwen3.6-35B-A3B-AWQ model weights
- 4x V100 16GB GPUs (or 4x A100 40GB recommended)
- Turbomind C++ extension compiled

Usage:
    python tests/test_lmdeploy/test_turbomind/test_ep4_model_loading.py
"""

import os
import sys
import time
from pathlib import Path

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_ep4_model_loading(model_path: str = None):
    """Test model loading with EP=4 configuration.

    Args:
        model_path: Path to Qwen3.6-35B-A3B-AWQ model
    """
    if model_path is None:
        model_path = os.environ.get('QWEN36_A3B_AWQ_PATH', '/data/models/Qwen3.6-35B-A3B-AWQ')

    if not os.path.exists(model_path):
        print(f"⚠️  Model path not found: {model_path}")
        print("   Set QWEN36_A3B_AWQ_PATH environment variable or provide model_path argument")
        print("   Skipping model loading test...")
        return

    print("=" * 70)
    print("EP=4 Model Loading Test")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Configuration: EP=4, TP=1, KV Quant=TurboQuant")
    print()

    try:
        from lmdeploy import TurbomindEngineConfig, pipeline
        from lmdeploy.turbomind import update_parallel_config

        # Create engine config with EP=4
        engine_config = TurbomindEngineConfig(
            ep=4,
            tp=1,
            device_num=4,
            session_len=2048,
            max_batch_size=1,
            quant_policy=8,  # KV cache quantization: K=4bit, V=2bit
            cache_max_entry_count=0.8,
        )

        print("Step 1: Creating pipeline with EP=4 configuration...")
        start_time = time.time()

        pipe = pipeline(
            model_path,
            backend='turbomind',
            engine_config=engine_config,
        )

        load_time = time.time() - start_time
        print(f"✅ Pipeline created successfully ({load_time:.2f}s)")
        print()

        # Test basic inference
        print("Step 2: Testing basic inference...")
        prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}: {prompt}")
            print("-" * 70)

            start_time = time.time()
            response = pipe([prompt])
            gen_time = time.time() - start_time

            output_text = response[0]
            print(f"Output: {output_text}")
            print(f"Generation time: {gen_time:.2f}s")

            # Basic quality checks
            if not output_text or len(output_text.strip()) == 0:
                print("❌ FAIL: Empty output")
                return False

            # Check for garbled output (e.g., all '!' characters)
            if output_text.count('!') > len(output_text) * 0.5:
                print("❌ FAIL: Output appears garbled (too many '!' characters)")
                return False

            # Check for excessive repetition
            words = output_text.split()
            if len(words) > 10:
                unique_words = set(words)
                if len(unique_words) < len(words) * 0.3:
                    print("⚠️  Warning: Output has high repetition")

            print("✅ PASS")

        print()
        print("=" * 70)
        print("🎉 All tests passed!")
        print("=" * 70)
        return True

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Test failed with error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


def test_ep4_vs_tp4_comparison(model_path: str = None):
    """Compare EP=4 vs TP=4 configurations.

    Args:
        model_path: Path to Qwen3.6-35B-A3B-AWQ model
    """
    if model_path is None:
        model_path = os.environ.get('QWEN36_A3B_AWQ_PATH', '/data/models/Qwen3.6-35B-A3B-AWQ')

    if not os.path.exists(model_path):
        print(f"⚠️  Model path not found: {model_path}")
        print("   Skipping EP=4 vs TP=4 comparison test...")
        return

    print("=" * 70)
    print("EP=4 vs TP=4 Comparison Test")
    print("=" * 70)
    print(f"Model: {model_path}")
    print()

    try:
        from lmdeploy import TurbomindEngineConfig, pipeline

        prompt = "Write a short poem about artificial intelligence."

        # Test EP=4
        print("Testing EP=4 configuration...")
        ep4_config = TurbomindEngineConfig(
            ep=4,
            tp=1,
            device_num=4,
            session_len=2048,
            max_batch_size=1,
            quant_policy=8,
        )

        ep4_pipe = pipeline(model_path, backend='turbomind', engine_config=ep4_config)

        start_time = time.time()
        ep4_response = ep4_pipe([prompt])
        ep4_time = time.time() - start_time
        ep4_output = ep4_response[0]

        print(f"EP=4 Output: {ep4_output}")
        print(f"EP=4 Time: {ep4_time:.2f}s")
        print()

        # Test TP=4
        print("Testing TP=4 configuration...")
        tp4_config = TurbomindEngineConfig(
            ep=1,
            tp=4,
            device_num=4,
            session_len=2048,
            max_batch_size=1,
            quant_policy=8,
        )

        tp4_pipe = pipeline(model_path, backend='turbomind', engine_config=tp4_config)

        start_time = time.time()
        tp4_response = tp4_pipe([prompt])
        tp4_time = time.time() - start_time
        tp4_output = tp4_response[0]

        print(f"TP=4 Output: {tp4_output}")
        print(f"TP=4 Time: {tp4_time:.2f}s")
        print()

        # Compare results
        print("=" * 70)
        print("Comparison Results:")
        print("=" * 70)
        print(f"EP=4 Time: {ep4_time:.2f}s")
        print(f"TP=4 Time: {tp4_time:.2f}s")
        print(f"Speedup: {tp4_time / ep4_time:.2f}x")
        print()

        # Quality comparison
        ep4_words = len(ep4_output.split())
        tp4_words = len(tp4_output.split())
        print(f"EP=4 Output Length: {ep4_words} words")
        print(f"TP=4 Output Length: {tp4_words} words")

        # Check for garbled output in EP=4
        if ep4_output.count('!') > len(ep4_output) * 0.5:
            print("❌ EP=4 output appears garbled")
        else:
            print("✅ EP=4 output quality looks good")

        if tp4_output.count('!') > len(tp4_output) * 0.5:
            print("❌ TP=4 output appears garbled")
        else:
            print("✅ TP=4 output quality looks good")

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run EP=4 model loading tests."""
    import argparse

    parser = argparse.ArgumentParser(description='EP=4 Model Loading Test')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to Qwen3.6-35B-A3B-AWQ model')
    parser.add_argument('--compare', action='store_true',
                        help='Run EP=4 vs TP=4 comparison test')
    args = parser.parse_args()

    if args.compare:
        test_ep4_vs_tp4_comparison(args.model_path)
    else:
        test_ep4_model_loading(args.model_path)


if __name__ == '__main__':
    main()
