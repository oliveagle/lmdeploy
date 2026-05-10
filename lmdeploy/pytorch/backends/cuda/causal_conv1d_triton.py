# Copyright (c) OpenMMLab. All rights reserved.
"""Triton fallback implementation of causal_conv1d for sm_70 (V100) and other GPUs where TileLang fails."""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def causal_conv1d_update_kernel(
        X, Conv_State, W, Bias, Out, Cache_seqlens, Conv_state_indices,
        batch, hidden_size, seqlen, state_len, width,
        is_circular_buffer: tl.constexpr, has_bias: tl.constexpr, has_state_indices: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton kernel for causal_conv1d_update."""
        idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < hidden_size
        idx = tl.where(mask, idx, 0)

        # Determine batch index
        batch_idx = tl.program_id(1)
        state_batch_idx = batch_idx
        if has_state_indices:
            state_batch_idx = tl.load(Conv_state_indices + batch_idx)

        # Load weight and bias
        if has_bias:
            b_val = tl.load(Bias + idx, mask=mask).to(tl.float32)
        else:
            b_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        w_vals = tl.load(W + idx[:, None] * width + tl.arange(0, width)[None, :], mask=mask[:, None])
        w_vals = w_vals.to(tl.float32)

        # Load initial state
        x_vals = tl.zeros([BLOCK_SIZE, width], dtype=tl.float32)
        if is_circular_buffer:
            cache_seqlen = tl.load(Cache_seqlens + batch_idx)
            update_idx = (cache_seqlen - (width - 1)) % state_len
        else:
            update_idx = state_len - (width - 1)

        for i in range(width - 1):
            if is_circular_buffer:
                current_idx = (update_idx + i) % state_len
            else:
                current_idx = update_idx + i

            state_pos = state_batch_idx * hidden_size * state_len + idx * state_len + current_idx
            state_val = tl.load(Conv_State + state_pos, mask=mask).to(tl.float32)
            x_vals[:, i] = state_val

        # Process each sequence position
        for i in range(seqlen):
            # Load input
            x_val = tl.load(X + batch_idx * hidden_size * seqlen + idx * seqlen + i, mask=mask).to(tl.float32)
            x_vals[:, width - 1] = x_val

            # Write to state
            if is_circular_buffer:
                write_idx = (update_idx + width - 1 + i) % state_len
            else:
                write_idx = update_idx + width - 1 + i
            write_pos = state_batch_idx * hidden_size * state_len + idx * state_len + write_idx
            tl.store(Conv_State + write_pos, x_val.to(Conv_State.dtype.element_ty), mask=mask)

            # Compute output
            out_val = b_val
            for j in range(width):
                out_val = out_val + w_vals[:, j] * x_vals[:, j]

            out_pos = batch_idx * hidden_size * seqlen + idx * seqlen + i
            tl.store(Out + out_pos, out_val.to(Out.dtype.element_ty), mask=mask)

            # Shift x_vals
            for j in range(width - 1):
                x_vals[:, j] = x_vals[:, j + 1]


class CausalConv1dTritonImpl:
    """CausalConv1d implementation using Triton for compatibility with older GPUs."""

    def conv1d_fn(self,
                  x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  seq_idx: torch.Tensor | None = None,
                  initial_states: torch.Tensor | None = None,
                  return_final_states: bool = False,
                  activation: str | None = None):
        """Forward pass - fallback to PyTorch since it's used less frequently."""
        assert x.dim() == 3, 'x should be in shape of [batch_size, hidden_size, sum_seqlen]'
        assert x.size(0) == 1, 'batch_size should be 1 for continuous batching'
        assert weight.dim() == 2, 'weight should be in shape of [hidden_size, kernel_size]'
        assert activation in ['silu', 'swish', None]

        # Pure PyTorch fallback for conv1d_fn
        batch_size, hidden_size, seqlen = x.shape
        kernel_size = weight.size(1)

        # Use unfold to simulate causal convolution
        # Pad input for causal convolution
        x_pad = x.permute(0, 2, 1).contiguous()  # [1, seqlen, hidden_size]
        x_pad = x_pad.view(1, seqlen, hidden_size)

        # Convolution using matrix multiply
        out = torch.zeros_like(x)

        for i in range(seqlen):
            start = max(0, i - kernel_size + 1)
            if seq_idx is not None:
                # TODO: handle seq_idx properly
                pass

            window = x[:, :, start:i+1]
            if window.size(2) < kernel_size:
                pad_size = kernel_size - window.size(2)
                pad = torch.zeros(1, hidden_size, pad_size, device=x.device, dtype=x.dtype)
                window = torch.cat([pad, window], dim=2)

            # Compute output
            o = (window * weight.unsqueeze(0)).sum(dim=2)
            if bias is not None:
                o = o + bias.unsqueeze(0)
            out[:, :, i] = o

        if activation in ['silu', 'swish']:
            out = out.sigmoid() * out

        if return_final_states:
            return out, None
        return out

    def update_fn(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
    ):
        """Update conv state using Triton."""
        # Handle input shape
        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)

        batch, hidden_size, seqlen = x.shape
        width = weight.size(-1)
        state_len = conv_state.size(-1)

        # Create output
        out = torch.empty_like(x)

        if HAS_TRITON:
            # Use Triton kernel
            BLOCK_SIZE = 64
            num_blocks = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE

            grid = (num_blocks, batch)

            causal_conv1d_update_kernel[grid](
                x, conv_state, weight, bias, out, cache_seqlens, conv_state_indices,
                batch, hidden_size, seqlen, state_len, width,
                is_circular_buffer=cache_seqlens is not None,
                has_bias=bias is not None,
                has_state_indices=conv_state_indices is not None,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            # Fallback to pure PyTorch
            for b in range(batch):
                if conv_state_indices is not None:
                    state_b = int(conv_state_indices[b].item())
                else:
                    state_b = b

                for h in range(hidden_size):
                    w = weight[h]
                    b_val = bias[h] if bias is not None else 0

                    # Load state
                    if cache_seqlens is not None:
                        cache_seqlen = int(cache_seqlens[b].item())
                        start = (cache_seqlen - (width - 1)) % state_len
                    else:
                        start = state_len - (width - 1)

                    state_vals = []
                    for i in range(width - 1):
                        if cache_seqlens is not None:
                            pos = (start + i) % state_len
                        else:
                            pos = start + i
                        state_vals.append(conv_state[state_b, h, pos].item())
                    state_vals = [float(v) for v in state_vals]

                    # Process each seq step
                    for i in range(seqlen):
                        x_val = x[b, h, i].item()
                        full_window = state_vals + [x_val]

                        # Compute output
                        o_val = b_val
                        for j in range(width):
                            o_val += w[j].item() * full_window[j]
                        out[b, h, i] = o_val

                        # Update state
                        if cache_seqlens is not None:
                            write_pos = (start + width - 1 + i) % state_len
                        else:
                            write_pos = start + width - 1 + i
                        conv_state[state_b, h, write_pos] = x_val

                        # Shift
                        state_vals = full_window[1:]

        # Apply activation
        if activation in ['silu', 'swish']:
            out = out.sigmoid() * out

        if unsqueeze:
            out = out.squeeze(-1)

        return out
