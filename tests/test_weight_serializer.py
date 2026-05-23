#!/usr/bin/env python3
"""
Quick test script for weight_serializer.py.
Tests with minimal data to verify functionality.
"""

import sys
import os
import tempfile
import struct

# Add lmdeploy to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lmdeploy.turbomind.weight_serializer import WeightSerializer, TensorInfo, WeightFile

def test_weight_file():
    """Test WeightFile tensor offset calculation."""
    print("[Test] WeightFile offset alignment...")

    import torch

    wf = WeightFile(path="test.bin")

    # Add tensors
    t1 = torch.zeros([100, 100], dtype=torch.float16)
    info1 = wf.add_tensor("tensor1", t1)
    assert info1.offset == 0, f"First tensor offset should be 0, got {info1.offset}"
    assert info1.size == 20000, f"Size should be 20000, got {info1.size}"

    # Second tensor should be at 64-byte aligned offset
    t2 = torch.zeros([50, 50], dtype=torch.float16)
    info2 = wf.add_tensor("tensor2", t2)
    expected_offset = ((20000 + 63) & ~63)  # 64-byte aligned
    assert info2.offset == expected_offset, f"Second tensor offset should be {expected_offset}, got {info2.offset}"

    print("[Test] PASSED: WeightFile offset alignment")


def test_serializer_config():
    """Test WeightSerializer config loading."""
    print("[Test] WeightSerializer config loading...")

    model_path = "/mnt/data/models/modelscope_models/Qwen3.6-35B-A3B-AWQ"

    if not os.path.exists(model_path):
        print(f"[Test] SKIPPED: Model path {model_path} not found")
        return

    ws = WeightSerializer(model_path, "/tmp/test_output")

    # Check loaded config (Qwen3.5 MoE has nested text_config)
    assert ws.hidden_size == 2048, f"hidden_size should be 2048, got {ws.hidden_size}"
    assert ws.num_layers == 40, f"num_layers should be 40, got {ws.num_layers}"
    assert ws.num_heads == 16, f"num_heads should be 16, got {ws.num_heads}"
    assert ws.num_kv_heads == 2, f"num_kv_heads should be 2, got {ws.num_kv_heads}"
    assert ws.is_awq == True, f"is_awq should be True, got {ws.is_awq}"

    print("[Test] PASSED: WeightSerializer config loading")


def test_serialize_single_file():
    """Test serialization with a minimal single safetensors file."""
    print("[Test] Minimal serialization test...")

    # This test requires creating a minimal safetensors file
    # For now, we'll skip the full serialization test

    print("[Test] SKIPPED: Full serialization (requires safetensors I/O)")


if __name__ == "__main__":
    print("=" * 60)
    print("Weight Serializer Tests")
    print("=" * 60)

    try:
        test_weight_file()
        test_serializer_config()
        test_serialize_single_file()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
