# DFlash Debugging Summary

## Issue Analysis

The original issue reported was that `enable_dflash_=0` during `UnifiedDecoder::Forward`, even after `EnableDFlash(true)` was called.

## Key Findings

### 1. Warmup vs. Actual Inference
The logs showing `enable_dflash_=0` were from **warmup phase**, not actual inference. The execution order is:

1. `_create_engine()` - Creates Engine and runs warmup
2. Warmup runs with `enable_dflash_=0` (default value)
3. `_load_dflash_model()` - Loads draft weights
4. `EnableDFlash(true)` - Sets `enable_dflash_=1`
5. **Actual inference** - Should run with `enable_dflash_=1`

### 2. Decoder Address Consistency
The decoder address is consistent throughout:
- During EnableDFlash: `decoder=0x...`
- During Forward: `decoder=0x...` (same address)

There are NO multiple decoder instances or copies. The issue was NOT about different decoder objects.

### 3. EnableDFlash Flow Works Correctly
From the logs:
```
[TM][INFO][0514.03:52:22.232745][unified_decoder.h:32] [DFlash] UnifiedDecoder::EnableDFlash: this=0x..., enable=true, enable_dflash_ before=0
[TM][INFO][0514.03:52:22.232746][unified_decoder.h:35] [DFlash] UnifiedDecoder::EnableDFlash: enable_dflash_ after=1
```

This confirms that:
- `EnableDFlash(true)` is being called
- `enable_dflash_` is being set from 0 to 1
- The setting persists (no immediate reset)

### 4. Memory and Architecture

**Engine Model Ownership:**
```
TurboMind::Impl::engines_[index] (vector<Engine>)
  └─ Engine::impl_ (unique_ptr<Impl>)
      └─ Engine::Impl::model_ (LanguageModel by value)
          └─ LanguageModel::impl_ (unique_ptr<Impl>)
              └─ LanguageModel::Impl::unified_decoder_ (unique_ptr<UnifiedDecoder>)
                  └─ UnifiedDecoder::enable_dflash_ (bool)
```

**ModelExecutor Reference:**
```
Engine::Impl::executor_ (ModelExecutor)
  └─ ModelExecutor::impl_->model_ (LanguageModel& reference)
```

ModelExecutor holds a **reference** to the same LanguageModel, so changes are visible.

## Current Status

1. ✅ DFlash weights are loaded successfully
2. ✅ DFlashDraftModel is created and attached
3. ✅ EnableDFlash sets `enable_dflash_=1`
4. ❓ Need to verify actual inference uses `enable_dflash_=1`

## Next Steps

To verify DFlash works during actual inference:

1. **Run actual inference** (not just warmup) and check logs for `enable_dflash_=1`
2. **Check if aux_hidden_states are collected** during Forward
3. **Verify draft generation** happens in GenerateDraft
4. **Check accept rate** from VerifyDraft

## Testing Commands

```python
from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig, SpeculativeConfig

spec_config = SpeculativeConfig(
    method='dflash',
    model='/path/to/draft/model',
    num_speculative_tokens=8,
)

tm_config = TurbomindEngineConfig(
    model_name='qwen2',
    tp=1,
    session_len=256,
    max_batch_size=1,
    speculative_config=spec_config,
)

tm = TurboMind.from_pretrained(model_path, engine_config=tm_config)
# Use tm.stream_infer() for actual inference
```

## Debug Logging Added

The following logging has been added to track DFlash state:

1. **unified_decoder.h**: EnableDFlash logs `enable_dflash_` before/after
2. **unified_decoder.cc**: Forward logs decoder address and `enable_dflash_`
3. **language_model.cc**: Forward logs LanguageModel address and decoder address
4. **turbomind.cc**: EnableDFlash logs engine size and validity

These logs help track the exact decoder object and its state throughout execution.
