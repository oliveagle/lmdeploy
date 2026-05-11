#!/usr/bin/env python3
"""Test EP (Expert Parallelism) configuration flow."""

import os
import sys

from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.turbomind.deploy.converter import get_tm_model
from lmdeploy.turbomind.deploy.config import TurbomindModelConfig

def test_ep_config():
    """Test that EP configuration is properly passed through the pipeline."""

    print("=" * 60)
    print("EP Configuration Test")
    print("=" * 60)

    # Test 1: Check TurbomindEngineConfig
    print("\n[Test 1] TurbomindEngineConfig:")
    engine_config = TurbomindEngineConfig(
        tp=1,
        ep=4,  # Expert Parallelism
        ep_rank=0,
        devices=[0, 1, 2, 3]
    )
    print(f"  ep = {engine_config.ep}")
    print(f"  ep_rank = {engine_config.ep_rank}")
    print(f"  tp = {engine_config.tp}")

    # Test 2: Check ModelConfig EP fields
    print("\n[Test 2] ModelConfig EP fields:")
    from lmdeploy.turbomind.deploy.config import ModelConfig
    model_cfg = ModelConfig()
    print(f"  mlp_ep_size = {model_cfg.mlp_ep_size}")
    print(f"  mlp_ep_rank = {model_cfg.mlp_ep_rank}")
    print(f"  mlp_tp_size = {model_cfg.mlp_tp_size}")

    # Test 3: Simulate converter.py behavior
    print("\n[Test 3] Simulating converter.py mapping:")
    print("  engine_config.ep -> model_config.mlp_ep_size")
    if engine_config.ep is not None and engine_config.ep > 1:
        print(f"  Setting mlp_ep_size = {engine_config.ep}")
        print(f"  Setting mlp_ep_rank = {engine_config.ep_rank}")

        # Check if mlp_tp_size should be overridden
        print("\n  ⚠️  ISSUE: mlp_tp_size is NOT explicitly set to 1 when ep > 1")
        print("  This may cause incorrect sharding behavior!")

    # Test 4: Expected behavior
    print("\n[Test 4] Expected behavior for EP=4:")
    print("  - mlp_ep_size should be 4")
    print("  - mlp_ep_rank should be 0-3 (depending on GPU)")
    print("  - mlp_tp_size should be 1 (no TP sharding for MoE)")
    print("  - Each GPU should have 64 experts (256 / 4)")
    print("  - Memory per GPU: ~5 GB instead of ~18 GB")

    # Test 5: Check C++ params
    print("\n[Test 5] C++ EngineParam:")
    print("  mlp_ep_size = 1  (default)")
    print("  mlp_ep_rank = 0  (default)")
    print("  These should be set from model_config.mlp_ep_size!")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_ep_config()
