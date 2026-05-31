# DFlash Speculative Deciving - Attribution and License

## Overview

DFlash (Diffusion Flash) speculative decoding is integrated into LMDeploy TurboMind to accelerate inference by generating draft tokens using a smaller model.

## License

```
Apache-2.0 WITH LLVM-exception
```

## Source Attribution

This implementation is derived from and inspired by:

- **lucebox-hub/dflash** (https://github.com/lucebox-hub/dflash)
  - Original authors: lucebox-hub contributors
  - License: Apache-2.0

## LMDeploy TurboMind Integration

The following modifications were made to adapt DFlash for LMDeploy TurboMind:

### Architecture Changes
- Integrated with TurboMind's memory management and tensor system
- Adapted to work with `UnifiedDecoder` for layer-wise hidden state collection
- Added DDTree (Dense Depth Tree) verification algorithm
- Implemented prefix caching for improved performance

### File Structure
```
src/turbomind/models/llama/
├── DFlashDraftModel.{h,cu}     # Main draft model implementation
├── DFlashDraftWeight.{h,cc}    # Weight structure and management
├── dflash_kernels.{h,cu}       # CUDA kernels (attention, etc.)
├── unified_decoder_dflash.h    # UnifiedDecoder integration
├── ddtree.{h,cpp}              # DDTree verification (original)
└── unified_decoder.cc          # Modified for DFlash support
```

### Key Differences from lucebox-hub/dflash
1. **Tensor System**: Uses TurboMind's `Tensor` class instead of PyTorch tensors
2. **Memory Management**: Integrated with TurboMind's allocator and stream system
3. **Decoder Integration**: Works with `UnifiedDecoder` rather than standalone
4. **Verification**: Uses custom DDTree implementation for parallel verification

## Performance

Expected performance improvements:
- **1.7x+** decode speedup
- **60-80%** draft token acceptance rate
- **~30%** additional speedup from DDTree parallel verification

## References

- Original lucebox-hub/dflash: https://github.com/lucebox-hub/dflash
- LMDeploy: https://github.com/InternLM/lmdeploy

---

*Last updated: 2026-05-12*
