# Copyright (c) OpenMMLab. All rights reserved.
"""Core CUDA kernel warmup functions for JIT compilation and kernel preheating.

These functions are registered with WarmupManager during import and executed
during the model warmup phase to:
1. Trigger Triton JIT compilation for key kernels
2. Warm up CUDA context and cuBLAS/cuBLASLt libraries
3. Preheat memory allocators
"""

import torch

from lmdeploy.pytorch.backends.cuda.warmup_manager import WarmupMeta, get_warmup_manager
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _warmup_core_ops(warmup_meta: WarmupMeta):
    """Warm up core CUDA operations: memory alloc, copy, and basic math."""
    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    max_batch = warmup_meta.max_batch_size

    # Allocate warmup buffers to preheat the CUDA memory allocator
    buf = torch.empty(max_tokens, max_tokens, dtype=dtype, device=device)
    buf.copy_(torch.randn_like(buf))

    # Basic math ops to warm up cuBLAS
    out = torch.mm(buf[:max_batch], buf[:max_batch, :max_batch])
    out += 1.0

    # Synchronize to ensure all ops complete
    torch.cuda.synchronize()


def _warmup_flash_attention(warmup_meta: WarmupMeta):
    """Warm up flash attention kernels for various shapes."""
    try:
        from lmdeploy.pytorch.kernels.cuda import flash_attn_varlen_func
    except ImportError:
        logger.debug('flash_attn_varlen_func not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    num_heads = warmup_meta.num_attention_heads
    num_kv_heads = warmup_meta.num_kv_heads
    head_dim = warmup_meta.head_dim

    # Warm up prefill attention with various seq lens
    for seq_len in [64, 256, 1024, min(max_tokens, 2048)]:
        q = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        q_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        q_seqlens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        kv_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        kv_seqlens = torch.tensor([seq_len], dtype=torch.int32, device=device)

        flash_attn_varlen_func(
            q, k, v,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,
            max_seqlen_q=seq_len,
        )

    torch.cuda.synchronize()


def _warmup_paged_attention(warmup_meta: WarmupMeta):
    """Warm up paged attention kernels for decoding."""
    try:
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache
    except ImportError:
        logger.debug('flash_attn_with_kvcache not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_batch = warmup_meta.max_batch_size
    num_heads = warmup_meta.num_attention_heads
    num_kv_heads = warmup_meta.num_kv_heads
    head_dim = warmup_meta.head_dim
    block_size = warmup_meta.block_size

    # Warm up decoding attention with various batch sizes
    for batch_size in [1, min(max_batch, 16)]:
        seq_len = 1  # decoding: one token per seq
        q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)

        # KV cache in bshd layout: (num_blocks, block_size, num_heads, head_dim)
        num_blocks = batch_size * 2
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)

        cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        # page_table: maps each sequence position to block index
        page_table = torch.arange(batch_size * seq_len, dtype=torch.int32, device=device).unsqueeze(0)
        page_table = page_table.repeat(batch_size, 1)[:batch_size, :seq_len]

        flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            kv_layout='bshd',
        )

    torch.cuda.synchronize()


def _warmup_rms_norm(warmup_meta: WarmupMeta):
    """Warm up RMS normalization kernel."""
    try:
        from lmdeploy.pytorch.kernels.cuda import rms_norm
    except ImportError:
        logger.debug('rms_norm module not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    hidden_dim = warmup_meta.hidden_size

    for num_tokens in [64, 256, min(max_tokens, 2048)]:
        hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        weight = torch.randn(hidden_dim, dtype=dtype, device=device)
        rms_norm(hidden_states, weight, 1e-5)

    torch.cuda.synchronize()


def _warmup_rotary_embedding(warmup_meta: WarmupMeta):
    """Warm up rotary positional embedding kernel."""
    try:
        from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb
    except ImportError:
        logger.debug('apply_rotary_pos_emb module not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    num_heads = warmup_meta.num_attention_heads
    head_dim = warmup_meta.head_dim

    for num_tokens in [64, 256, min(max_tokens, 2048)]:
        q = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
        cos = torch.randn(num_tokens, head_dim // 2, dtype=dtype, device=device)
        sin = torch.randn(num_tokens, head_dim // 2, dtype=dtype, device=device)
        position_ids_1d = torch.arange(num_tokens, dtype=torch.int64, device=device)

        apply_rotary_pos_emb(q, k, cos, sin, position_ids_1d, num_heads)

    torch.cuda.synchronize()


def _warmup_fill_kv_cache(warmup_meta: WarmupMeta):
    """Warm up KV cache filling kernels."""
    try:
        from lmdeploy.pytorch.kernels.cuda import fill_kv_cache
    except ImportError:
        logger.debug('fill_kv_cache module not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    num_heads = warmup_meta.num_attention_heads
    head_dim = warmup_meta.head_dim
    block_size = warmup_meta.block_size

    num_blocks = max(16, (max_tokens + block_size - 1) // block_size + 8)
    # bshd layout: (num_blocks, block_size, num_heads, head_dim)
    k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)

    k_states = torch.randn(max_tokens, num_heads, head_dim, dtype=dtype, device=device)
    v_states = torch.randn(max_tokens, num_heads, head_dim, dtype=dtype, device=device)

    q_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
    q_seq_length = torch.tensor([max_tokens], dtype=torch.int32, device=device)
    kv_seq_length = torch.tensor([max_tokens], dtype=torch.int32, device=device)
    block_offsets = torch.arange(max_tokens, dtype=torch.int32, device=device).unsqueeze(0)

    fill_kv_cache(
        k_states, v_states,
        k_cache, v_cache,
        q_start_loc, q_seq_length, kv_seq_length,
        max_q_seq_length=max_tokens,
        block_offsets=block_offsets,
        kv_layout='bshd',
    )
    torch.cuda.synchronize()


def _warmup_activation(warmup_meta: WarmupMeta):
    """Warm up activation kernels (silu_and_mul)."""
    try:
        from lmdeploy.pytorch.kernels.cuda import silu_and_mul
    except ImportError:
        logger.debug('silu_and_mul not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    intermediate_size = warmup_meta.intermediate_size

    for num_tokens in [64, 256, min(max_tokens, 2048)]:
        gate_up = torch.randn(num_tokens, 2 * intermediate_size, dtype=dtype, device=device)
        out = torch.empty(num_tokens, intermediate_size, dtype=dtype, device=device)
        silu_and_mul(out, gate_up)

    torch.cuda.synchronize()


def _warmup_awq_kernels(warmup_meta: WarmupMeta):
    """Warm up AWQ dequantization and GEMM kernels."""
    try:
        from lmdeploy.pytorch.kernels.cuda import awq_gemm, awq_dequantize
    except ImportError:
        logger.debug('awq_gemm not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    hidden_size = warmup_meta.hidden_size
    intermediate_size = warmup_meta.intermediate_size
    group_size = 128

    for num_tokens in [64, 256, min(max_tokens, 2048)]:
        for k, n in [(hidden_size, hidden_size), (hidden_size, intermediate_size)]:
            try:
                x = torch.randn(num_tokens, k, dtype=dtype, device=device)
                qweight = torch.randint(0, 256, (k // 16, n * 16 // 8), dtype=torch.int32, device=device)
                qzeros = torch.randint(0, 256, (k // group_size, n // 256 * 16), dtype=torch.int32, device=device)
                scales = torch.randn(k // group_size, n, dtype=dtype, device=device)
                out = torch.empty(num_tokens, n, dtype=dtype, device=device)
                awq_gemm(x, qweight, scales, qzeros, out, group_size=group_size, gemm_type=0)
            except Exception:
                pass

    torch.cuda.synchronize()


def _warmup_w8a8_gemm(warmup_meta: WarmupMeta):
    """Warm up W8A8 GEMM Triton kernel."""
    try:
        from lmdeploy.pytorch.kernels.cuda import w8a8_gemm
    except ImportError:
        logger.debug('w8a8_gemm not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    hidden_size = warmup_meta.hidden_size
    intermediate_size = warmup_meta.intermediate_size

    for num_tokens in [64, 256, min(max_tokens, 2048)]:
        for k, n in [(hidden_size, hidden_size), (hidden_size, intermediate_size)]:
            try:
                a = torch.randint(-128, 127, (num_tokens, k), dtype=torch.int8, device=device)
                b = torch.randint(-128, 127, (n, k), dtype=torch.int8, device=device)
                scale_a = torch.randn(num_tokens, dtype=dtype, device=device)
                scale_b = torch.randn(n, dtype=dtype, device=device)
                w8a8_gemm(a, scale_a, b, scale_b, num_tokens, n, k)
            except Exception:
                pass

    torch.cuda.synchronize()


def _warmup_fused_moe(warmup_meta: WarmupMeta):
    """Warm up fused MoE Triton kernel."""
    try:
        from lmdeploy.pytorch.kernels.cuda import fused_moe
    except ImportError:
        logger.debug('fused_moe not available, skipping warmup')
        return

    device = 'cuda'
    dtype = warmup_meta.dtype
    max_tokens = warmup_meta.max_num_tokens
    hidden_size = warmup_meta.hidden_size
    intermediate_size = warmup_meta.intermediate_size
    num_experts = 8

    for num_tokens in [64, 256, min(max_tokens, 2048)]:
        try:
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            w1 = torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
            w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
            topk_ids = torch.randint(0, num_experts, (num_tokens, 2), dtype=torch.int64, device=device)
            topk_weights = torch.randn(num_tokens, 2, dtype=dtype, device=device)
            fused_moe(x, w1, w2, topk_ids, topk_weights, False)
        except Exception:
            pass

    torch.cuda.synchronize()


def _warmup_multinomial_sampling(warmup_meta: WarmupMeta):
    """Warm up multinomial sampling kernel."""
    try:
        from lmdeploy.pytorch.kernels.cuda import multinomial_sampling
    except ImportError:
        logger.debug('multinomial_sampling module not available, skipping warmup')
        return

    device = 'cuda'
    max_batch = warmup_meta.max_batch_size
    vocab_size = warmup_meta.vocab_size

    for batch_size in [1, min(max_batch, 32)]:
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
        probs = torch.softmax(logits, dim=-1)
        indices = multinomial_sampling(probs, num_samples=1)
        # Force sync
        _ = indices.item()

    torch.cuda.synchronize()


def register_kernel_warmups():
    """Register all core kernel warmups with the WarmupManager.

    Each warmup is registered once (checked via `__contains__`) so that
    multiple calls to this function do not duplicate registrations.
    """
    warmup_mgr = get_warmup_manager()

    warmups = {
        'cuda_core_ops': _warmup_core_ops,
        'flash_attention': _warmup_flash_attention,
        'paged_attention': _warmup_paged_attention,
        'rms_norm': _warmup_rms_norm,
        'rotary_embedding': _warmup_rotary_embedding,
        'activation': _warmup_activation,
        'fill_kv_cache': _warmup_fill_kv_cache,
        'awq_kernels': _warmup_awq_kernels,
        'w8a8_gemm': _warmup_w8a8_gemm,
        'fused_moe': _warmup_fused_moe,
        'multinomial_sampling': _warmup_multinomial_sampling,
    }

    for name, func in warmups.items():
        if name not in warmup_mgr:
            warmup_mgr[name] = func
