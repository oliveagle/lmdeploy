# Copyright (c) OpenMMLab. All rights reserved.
"""Attention buffer cache for memory-efficient intermediate storage.

Reuses flatten_kv_cache output buffers and paged attention acc buffers
across layers within a forward pass, reducing peak memory from
O(num_layers * buffer_size) to O(buffer_size).

The cache is keyed by buffer shape signature (device, layout, num_heads,
seq_len, head_dim, dtype). When the shape matches a cached buffer, it
is reused; otherwise a new buffer is allocated and cached.

Buffers are released after each layer's forward pass, so they can be
safely reused by the next layer since layers execute sequentially.
"""

import threading

import torch

from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")

_lock = threading.Lock()
_cache: dict[tuple, torch.Tensor] = {}
_paged_acc_cache: dict[tuple, torch.Tensor] = {}
_enabled = True
_hits = 0
_misses = 0


def _make_key(device, buffer_type, num_heads, seq_len, head_dim, dtype, layout="hsd"):
    """Create hashable cache key."""
    return (str(device), buffer_type, num_heads, seq_len, head_dim, str(dtype), layout)


def _make_paged_acc_key(device, num_tokens, num_heads, split_k, last_dim, dtype):
    """Create hashable cache key for paged attention acc buffer."""
    return (str(device), num_tokens, num_heads, split_k, last_dim, str(dtype))


def get_or_allocate(
    device: torch.device, buffer_type: str, shape: tuple, dtype: torch.dtype, layout: str = "hsd"
) -> torch.Tensor:
    """Get a cached buffer or allocate a new one.

    Args:
        device: Target device.
        buffer_type: Type of buffer ('flatten_k', 'flatten_v').
        shape: Required buffer shape.
        dtype: Buffer dtype.
        layout: Memory layout ('hsd' or 'shd').

    Returns:
        Cached or newly allocated buffer tensor.
    """
    global _hits, _misses

    if not _enabled:
        return torch.empty(shape, dtype=dtype, device=device)

    num_heads = shape[0] if layout == "hsd" else shape[1]
    seq_len = shape[1] if layout == "hsd" else shape[0]
    head_dim = shape[2]

    key = _make_key(device, buffer_type, num_heads, seq_len, head_dim, dtype, layout)

    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached.shape == shape and cached.dtype == dtype and cached.device == device:
            _hits += 1
            return cached

        _misses += 1
        buf = torch.empty(shape, dtype=dtype, device=device)
        _cache[key] = buf
        return buf


def clear():
    """Clear all cached buffers."""
    global _hits, _misses
    with _lock:
        _cache.clear()
        _paged_acc_cache.clear()
        _hits = 0
        _misses = 0


def get_or_allocate_paged_acc(device: torch.device, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Get a cached paged attention acc buffer or allocate a new one.

    Args:
        device: Target device.
        shape: Required buffer shape (num_tokens, num_heads, split_k, last_dim).
        dtype: Buffer dtype (always float32 for acc).

    Returns:
        Cached or newly allocated buffer tensor.
    """
    global _hits, _misses

    if not _enabled:
        return torch.empty(shape, dtype=dtype, device=device)

    num_tokens, num_heads, split_k, last_dim = shape
    key = _make_paged_acc_key(device, num_tokens, num_heads, split_k, last_dim, dtype)

    with _lock:
        cached = _paged_acc_cache.get(key)
        if cached is not None and cached.shape == shape and cached.dtype == dtype and cached.device == device:
            _hits += 1
            return cached

        _misses += 1
        buf = torch.empty(shape, dtype=dtype, device=device)
        _paged_acc_cache[key] = buf
        return buf


def enable():
    """Enable buffer caching."""
    global _enabled
    _enabled = True


def disable():
    """Disable buffer caching and clear cache."""
    global _enabled
    _enabled = False
    clear()


def is_enabled() -> bool:
    """Check if caching is enabled."""
    return _enabled


def get_stats() -> dict:
    """Get cache statistics."""
    with _lock:
        all_buffers = list(_cache.values()) + list(_paged_acc_cache.values())
        total_memory = sum(t.numel() * t.element_size() for t in all_buffers)
        total = _hits + _misses
        return {
            "enabled": _enabled,
            "flatten_buffers": len(_cache),
            "paged_acc_buffers": len(_paged_acc_cache),
            "total_memory_mb": total_memory / (1024 * 1024),
            "hits": _hits,
            "misses": _misses,
            "hit_rate": _hits / max(1, total),
        }
