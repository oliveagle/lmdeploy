#!/usr/bin/env python3
"""Patch to disable TileLang and use Dao's causal_conv1d or fallback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Monkey patch to disable TileLang
def patch_tilelang():
    from lmdeploy.pytorch.backends.cuda.utils import has_tilelang as original_has_tilelang

    def has_tilelang_disabled():
        return False

    import lmdeploy.pytorch.backends.cuda.utils
    lmdeploy.pytorch.backends.cuda.utils.has_tilelang = has_tilelang_disabled
    print("Disabled TileLang")

    # Clear the builder cache
    import lmdeploy.pytorch.backends.cuda.causal_conv1d as causal_conv1d_module
    causal_conv1d_module.CausalConv1dCudaBuilder.build.cache_clear()
    print("Cleared builder cache")

if __name__ == '__main__':
    patch_tilelang()
