# Copyright (c) OpenMMLab. All rights reserved.
import random
import time
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.utils import singleton
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


@dataclass
class WarmupMeta:
    """Warmup configuration dataclass.

    Parameters that drive JIT kernel specialization and kernel preheat sizing.

    Attributes:
        max_num_tokens: Maximum prefill token count
        max_batch_size: Maximum concurrent batch size
        dtype: Model weight/data dtype
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of Q heads
        num_kv_heads: Number of KV heads (for GQA/MQA)
        head_dim: Q head dimension
        head_dim_v: V head dimension (may differ from head_dim)
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        block_size: KV cache block size (paged attention)
        vocab_size: Vocabulary size
        quant_config: Quantization type string, e.g. 'awq', 'gptq', ''
    """
    max_num_tokens: int
    max_batch_size: int
    dtype: torch.dtype
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    head_dim_v: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 11008
    block_size: int = 64
    vocab_size: int = 32000
    quant_config: str = ''


@singleton
class WarmupManager:
    """Manager for CUDA warmup operations including JIT compilation and kernel preheating.

    Warmup is split into two phases:
    1. Registration: Kernel modules register warmup functions during import/initialization.
    2. Execution: Once all warmups are registered, warmup() is called to execute them.

    Each registered warmup receives a unique key and is shuffled to avoid bias in
    compilation order.
    """

    def __init__(self):
        self._warmup_calls = dict()

    def __contains__(self, key: str):
        """Contain key."""
        return key in self._warmup_calls

    def __getitem__(self, key: str):
        """Get item."""
        return self._warmup_calls.get(key, None)

    def __setitem__(self, key: str, val):
        """Set item."""
        self._warmup_calls[key] = val

    def __len__(self):
        """Get number of registered warmup functions."""
        return len(self._warmup_calls)

    @staticmethod
    def _run_with_timing(name: str, func, warmup_meta):
        """Run a warmup function and log timing."""
        start = time.perf_counter()
        try:
            func(warmup_meta)
            elapsed = time.perf_counter() - start
            logger.debug(f'Warmup {name} completed in {elapsed:.3f}s')
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.warning(f'Warmup {name} failed after {elapsed:.3f}s: {e}')

    def warmup(self, warmup_meta: WarmupMeta):
        """Execute all registered warmup functions with timing.

        Warmup functions are shuffled to avoid bias in compilation order.
        """
        if len(self._warmup_calls) == 0:
            return
        logger.info(f'Warming up {len(self._warmup_calls)} kernel(s).')
        funcs = list(self._warmup_calls.items())
        random.shuffle(funcs)
        for name, func in funcs:
            self._run_with_timing(name, func, warmup_meta)

    def warmup_jit(self, warmup_meta: WarmupMeta):
        """Run JIT-only warmup with minimal inputs to trigger compilation.

        Uses small batch/token sizes so kernels compile quickly without
        consuming significant GPU time.
        """
        jit_meta = WarmupMeta(
            max_num_tokens=1,
            max_batch_size=1,
            dtype=warmup_meta.dtype,
            num_hidden_layers=warmup_meta.num_hidden_layers,
            num_attention_heads=warmup_meta.num_attention_heads,
            num_kv_heads=warmup_meta.num_kv_heads,
            head_dim=warmup_meta.head_dim,
            head_dim_v=warmup_meta.head_dim_v,
            hidden_size=warmup_meta.hidden_size,
            intermediate_size=warmup_meta.intermediate_size,
            block_size=warmup_meta.block_size,
            vocab_size=warmup_meta.vocab_size,
            quant_config=warmup_meta.quant_config,
        )

        if len(self._warmup_calls) == 0:
            return

        logger.info('Starting JIT warmup (minimal inputs).')
        funcs = list(self._warmup_calls.items())
        for name, func in funcs:
            self._run_with_timing(f'JIT:{name}', func, jit_meta)
        logger.info('JIT warmup complete.')

    def warmup_kernel(self, warmup_meta: WarmupMeta):
        """Run kernel preheat with production-size inputs.

        Uses full max_num_tokens / max_batch_size so that kernel caches
        (L2, shared memory paths) are warmed to production conditions.
        """
        if len(self._warmup_calls) == 0:
            return

        logger.info('Starting kernel preheat (production-size inputs).')
        funcs = list(self._warmup_calls.items())
        random.shuffle(funcs)
        for name, func in funcs:
            self._run_with_timing(f'Preheat:{name}', func, warmup_meta)
        logger.info('Kernel preheat complete.')

    def warmup_full(self, warmup_meta: WarmupMeta):
        """Run complete warmup: JIT compilation followed by kernel preheat.

        This is the recommended warmup sequence:
        1. JIT warmup with minimal inputs (fast compilation)
        2. Kernel preheat with production inputs (hardware warmup)
        """
        if len(self._warmup_calls) == 0:
            return

        logger.info('Starting full warmup (JIT + kernel preheat).')
        self.warmup_jit(warmup_meta)
        self.warmup_kernel(warmup_meta)
        logger.info('Full warmup complete.')


def get_warmup_manager():
    """Get warmup manager."""
    return WarmupManager()
