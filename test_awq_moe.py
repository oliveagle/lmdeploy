#!/usr/bin/env python3
"""Test script for AWQ MoE implementation."""

import sys
import torch

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    try:
        from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ, AwqLinearWeights
        from lmdeploy.pytorch.nn.moe import build_fused_moe
        from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_dequant_weights
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_awq_linear_weights():
    """Test AwqLinearWeights class."""
    print("\nTesting AwqLinearWeights...")
    try:
        from lmdeploy.pytorch.nn.moe.awq import AwqLinearWeights

        device = torch.device('cpu')  # Use CPU for testing
        num_experts = 4
        in_features = 128
        out_features = 256
        w_bit = 4
        group_size = 128

        weights = AwqLinearWeights(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            w_bit=w_bit,
            group_size=group_size,
            weight_type='gate_up',
            device=device,
            bias=False,
            expert_list=None,
        )

        # Check shapes
        assert weights.qweight.shape == (num_experts, in_features, out_features // 8), \
            f"qweight shape mismatch: {weights.qweight.shape}"
        assert weights.scales.shape == (num_experts, in_features // group_size, out_features), \
            f"scales shape mismatch: {weights.scales.shape}"
        assert weights.qzeros.shape == (num_experts, in_features // group_size, out_features // 8), \
            f"qzeros shape mismatch: {weights.qzeros.shape}"

        print(f"✓ AwqLinearWeights shapes correct:")
        print(f"  qweight: {weights.qweight.shape}")
        print(f"  scales: {weights.scales.shape}")
        print(f"  qzeros: {weights.qzeros.shape}")
        return True
    except Exception as e:
        print(f"✗ AwqLinearWeights test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_build_fused_moe_awq():
    """Test build_fused_moe with AWQ quantization."""
    print("\nTesting build_fused_moe with AWQ...")
    try:
        from lmdeploy.pytorch.nn.moe import build_fused_moe
        from lmdeploy.pytorch.config import QuantizationConfig, ModelConfig
        from lmdeploy.pytorch.models.patch import build_model_context, BuildModelContext

        device = torch.device('cpu')
        hidden_dim = 128
        ffn_dim = 256
        num_experts = 4
        top_k = 2

        # Create AWQ quantization config
        quant_config = QuantizationConfig(
            quant_method='awq',
            bits=4,
            group_size=128,
        )

        # Create BuildModelContext with quant_config
        ctx = BuildModelContext()
        ctx.quant_config = quant_config

        # Use build_model_context context manager
        with build_model_context(ctx):
            # Build FusedMoEAWQ
            moe = build_fused_moe(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=top_k,
                bias=False,
                renormalize=False,
                dtype=torch.float16,
                device=device,
                all_reduce=False,
                enable_ep=False,
                quant_config=quant_config,
                layer_idx=0,
                act_func=None,
                prefix='',
            )

            assert moe is not None, "build_fused_moe returned None"
            assert type(moe).__name__ == 'FusedMoEAWQ', \
                f"Expected FusedMoEAWQ, got {type(moe).__name__}"

            print(f"✓ build_fused_moe created FusedMoEAWQ successfully")
            print(f"  Type: {type(moe).__name__}")
            print(f"  num_experts: {moe.num_experts}")
            print(f"  w_bit: {moe.w_bit}")
            print(f"  group_size: {moe.group_size}")
            return True

    except Exception as e:
        print(f"✗ build_fused_moe test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fused_moe_awq_forward():
    """Test FusedMoEAWQ forward pass (shape only, no actual computation)."""
    print("\nTesting FusedMoEAWQ forward pass...")
    try:
        from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ
        from lmdeploy.pytorch.config import QuantizationConfig

        device = torch.device('cpu')
        hidden_dim = 128
        ffn_dim = 256
        num_experts = 4
        top_k = 2
        batch_size = 2
        seq_len = 8

        # Create FusedMoEAWQ
        moe = FusedMoEAWQ(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            w_bit=4,
            group_size=128,
            bias=False,
            renormalize=False,
            dtype=torch.float16,
            device=device,
            all_reduce=False,
            layer_idx=0,
            act_func=None,
        )

        # Create test inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
        topk_weights = torch.randn(batch_size, seq_len, top_k, dtype=torch.float16, device=device)
        topk_ids = torch.randint(0, num_experts, (batch_size, seq_len, top_k), device=device)

        print(f"  Input shapes:")
        print(f"    hidden_states: {hidden_states.shape}")
        print(f"    topk_weights: {topk_weights.shape}")
        print(f"    topk_ids: {topk_ids.shape}")

        # Note: We can't actually run forward on CPU because it requires CUDA
        # But we can verify the methods exist and have correct signatures
        assert hasattr(moe, 'dispatch'), "FusedMoEAWQ missing dispatch method"
        assert hasattr(moe, 'gemm'), "FusedMoEAWQ missing gemm method"
        assert hasattr(moe, 'combine'), "FusedMoEAWQ missing combine method"

        print(f"✓ FusedMoEAWQ has all required methods (dispatch, gemm, combine)")
        print(f"  Note: Actual forward pass requires CUDA")
        return True
    except Exception as e:
        print(f"✗ FusedMoEAWQ forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AWQ MoE Implementation Test Suite")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("AwqLinearWeights", test_awq_linear_weights()))
    results.append(("build_fused_moe AWQ", test_build_fused_moe_awq()))
    results.append(("FusedMoEAWQ forward", test_fused_moe_awq_forward()))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30s} {status}")

    all_passed = all(r[1] for r in results)

    print("=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
        return 0
    else:
        print("Some tests FAILED! ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
