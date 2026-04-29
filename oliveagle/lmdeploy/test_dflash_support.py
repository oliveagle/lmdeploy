#!/usr/bin/env python3
"""DFlash support test for lmdeploy.

This script tests the DFlash speculative decoding implementation.
"""

import os
import sys
import torch

# Add lmdeploy to path
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

from lmdeploy.pytorch.config import SpecDecodeConfig
from lmdeploy.pytorch.spec_decode.proposers.base import build_specdecode_proposer
from lmdeploy.pytorch.models.module_map import MODULE_MAP


def test_dflash_registration():
    """Test that DFlash is properly registered."""
    print("Testing DFlash registration...")

    # Check module map
    assert 'DFlashForCausalLM' in MODULE_MAP, "DFlashForCausalLM not in MODULE_MAP"
    print("✓ DFlashForCausalLM registered in MODULE_MAP")

    # Check proposer registry
    from lmdeploy.pytorch.spec_decode.proposers.base import SPEC_PROPOSERS
    assert 'dflash' in SPEC_PROPOSERS.module_dict, "dflash not in SPEC_PROPOSERS"
    print("✓ dflash proposer registered")

    print("All registration tests passed!")


def test_dflash_config():
    """Test DFlash config parsing."""
    print("\nTesting DFlash config...")

    # Create a mock SpecDecodeConfig
    try:
        config = SpecDecodeConfig(
            model='test-model',
            method='dflash',
            num_speculative_tokens=8,
        )
        print(f"✓ SpecDecodeConfig created with method='dflash'")
    except Exception as e:
        print(f"✗ Failed to create SpecDecodeConfig: {e}")
        return False

    return True


def test_dflash_imports():
    """Test that DFlash modules can be imported."""
    print("\nTesting DFlash imports...")

    try:
        from lmdeploy.pytorch.models.dflash import (
            DFlashAttention,
            DFlashDecoderLayer,
            DFlashModel,
            DFlashForCausalLM,
        )
        print("✓ DFlash model classes imported")
    except ImportError as e:
        print(f"✗ Failed to import DFlash model classes: {e}")
        return False

    try:
        from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlashProposer
        print("✓ DFlashProposer imported")
    except ImportError as e:
        print(f"✗ Failed to import DFlashProposer: {e}")
        return False

    return True


def test_dflash_attention():
    """Test DFlashAttention forward pass."""
    print("\nTesting DFlashAttention...")

    try:
        from lmdeploy.pytorch.models.dflash import DFlashAttention
        from transformers import PretrainedConfig

        # Create a minimal config
        config = PretrainedConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            attention_bias=False,
        )

        attn = DFlashAttention(config, dtype=torch.float16, device='cpu')

        # Test forward pass
        batch_seq_len = 4
        hidden_size = 128
        hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.float16)
        target_hidden = torch.randn(batch_seq_len, hidden_size, dtype=torch.float16)

        # Create dummy rotary_pos_emb
        cos = torch.randn(batch_seq_len, 64, dtype=torch.float16)
        sin = torch.randn(batch_seq_len, 64, dtype=torch.float16)
        rotary_pos_emb = (cos, sin)

        output = attn(hidden_states, target_hidden, rotary_pos_emb)

        assert output.shape == hidden_states.shape, f"Output shape mismatch: {output.shape} vs {hidden_states.shape}"
        print(f"✓ DFlashAttention forward pass successful, output shape: {output.shape}")

    except Exception as e:
        print(f"✗ DFlashAttention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_dflash_no_cache_config():
    """Test that dflash is in no_caches list."""
    print("\nTesting DFlash no-cache config...")

    # Read config.py to verify dflash is in no_caches
    import inspect
    from lmdeploy.pytorch import config

    source = inspect.getsource(config.SpecDecodeConfig.from_config)
    assert 'dflash' in source and 'no_caches' in source, "dflash not found in no_caches"
    print("✓ dflash is in no_caches list (no KV cache needed)")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DFlash Support Test Suite for lmdeploy")
    print("=" * 60)

    tests = [
        test_dflash_registration,
        test_dflash_config,
        test_dflash_imports,
        test_dflash_attention,
        test_dflash_no_cache_config,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
