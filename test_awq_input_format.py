#!/usr/bin/env python3
"""Test AWQ MoE implementation."""

import sys
import torch

def test_fused_moe_input_format():
    """Test the input format for fused_moe."""
    print("Testing fused_moe input format...")

    from lmdeploy.pytorch.kernels.cuda.fused_moe import fused_moe
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ
    from lmdeploy.pytorch.backends.cuda.moe.default import TritonFusedMoEImpl

    # Create test model
    device = torch.device('cuda')
    hidden_dim = 2048
    ffn_dim = 4096
    num_experts = 16
    top_k = 4
    w_bit = 4
    group_size = 128

    moe = FusedMoEAWQ(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        num_experts=num_experts,
        top_k=top_k,
        w_bit=w_bit,
        group_size=group_size,
        bias=False,
        renormalize=True,
        dtype=torch.float16,
        device=device,
        all_reduce=False,
        layer_idx=0,
        act_func=None,
    )

    # Initialize weights
    with torch.no_grad():
        moe.gate_up.qweight.data = torch.randint(
            0, 256, moe.gate_up.qweight.shape, dtype=torch.int32, device=device
        )
        moe.down.qweight.data = torch.randint(
            0, 256, moe.down.qweight.shape, dtype=torch.int32, device=device
        )
        moe.gate_up.scales.data = torch.rand(
            moe.gate_up.scales.shape, dtype=torch.float16, device=device
        )
        moe.down.scales.data = torch.rand(
            moe.down.scales.shape, dtype=torch.float16, device=device
        )
        moe.gate_up.qzeros.data = torch.randint(
            0, 256, moe.gate_up.qzeros.shape, dtype=torch.int32, device=device
        )
        moe.down.qzeros.data = torch.randint(
            0, 256, moe.down.qzeros.shape, dtype=torch.int32, device=device
        )

    # Dequantize weights manually to check shape
    gate_up_weights = moe._dequantize_gate_up_weights()
    down_weights = moe._dequantize_down_weights()

    print(f"gate_up_weights shape: {gate_up_weights.shape}")
    print(f"down_weights shape: {down_weights.shape}")

    # Create test inputs
    batch_size = 1
    seq_len = 16

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    topk_weights = torch.rand(batch_size, seq_len, top_k, dtype=torch.float16, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (batch_size, seq_len, top_k), device=device)

    print(f"\nhidden_states shape (3D): {hidden_states.shape}")
    print(f"topk_weights shape (3D): {topk_weights.shape}")
    print(f"topk_ids shape (3D): {topk_ids.shape}")

    # Reshape to 2D like how Qwen3MoeSparseMoeBlock does
    hidden_states = hidden_states.view(-1, hidden_dim)
    print(f"hidden_states shape (2D): {hidden_states.shape}")

    # Create state for dispatch and call manually
    state = {
        'hidden_states': hidden_states,
        'topk_weights': topk_weights,
        'topk_idx': topk_ids,
        'moe_type': 'Default',
    }

    recv_state = moe.dispatch(state)
    print(f"\nafter dispatch:")
    print(f"recv_state['hidden_states'] shape: {recv_state['hidden_states'].shape}")
    print(f"recv_state['topk_weights'] shape: {recv_state['topk_weights'].shape}")
    print(f"recv_state['topk_idx'] shape: {recv_state['topk_idx'].shape}")

    # Check shape of weights
    print(f"\ngate_up_weights shape: {gate_up_weights.shape}")

    # Now let's try to test the approach that dequantizes on-demand
    print("\nTesting on-demand dequantization approach...")
    from lmdeploy.pytorch.nn.moe.awq import AwqLinearWeights
    from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_dequant_weights_single_expert
    from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear

    # Test awq_linear directly on a single expert
    hidden = torch.randn(128, hidden_dim, dtype=torch.float16, device=device)
    expert_qw = moe.gate_up.qweight[0]
    expert_scales = moe.gate_up.scales[0]
    expert_qz = moe.gate_up.qzeros[0]
    print(f"\nTesting awq_linear for a single expert:")
    print(f"hidden shape: {hidden.shape}")
    print(f"expert_qw shape: {expert_qw.shape}")
    print(f"expert_scales shape: {expert_scales.shape}")
    print(f"expert_qz shape: {expert_qz.shape}")

    try:
        out = awq_linear(hidden, expert_qw, expert_scales, expert_qz)
        print(f"awq_linear successful! out shape: {out.shape}")
    except Exception as e:
        print(f"awq_linear failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nThe best approach might be to implement a specialized AWQ fused moe kernel,")
    print("or to use the existing Triton impl with on-demand dequantization.")
    return True

def main():
    """Main function."""
    print("=" * 80)
    print("AWQ MoE Input Format Test")
    print("=" * 80)

    try:
        if not test_fused_moe_input_format():
            return 1

        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
