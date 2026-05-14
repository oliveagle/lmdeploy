# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

### DFlash Speculative Decoding Log Pattern

All DFlash-related logs use the `[DFlash]` prefix for easy filtering:
- `unified_decoder.cc`: DECODE/PREFILL mode transitions, draft token verification
- `generation.cc`: Accepted tokens usage, statistics tracking
- `language_model.cc`: EnableDFlash initialization, draft model creation
- `turbomind.cc`: Weight loading, engine enablement

### TensorMap args for DFlash

Key tensors passed between modules via `args`:
- `dflash_stored_draft_tokens`: Draft tokens from previous iteration (Tensor)
- `dflash_accepted_tokens`: Verified tokens to output (Tensor)
- `dflash_accept_mask`: Accept/reject mask (Tensor)
- `logits`: Target model logits for verification (Tensor)

### DFlash Enablement Flow

1. Python: `SpeculativeConfig(method='dflash')` → `turbomind.py`
2. `turbomind.py`: `_load_dflash()` loads weights, calls `enable_dflash()`
3. C++: `TurboMind::EnableDFlash` → `Engine::EnableDFlash` → `LanguageModel::EnableDFlash`
4. `LanguageModel::EnableDFlash`: Creates `DFlashDraftModel`, sets on `UnifiedDecoder`
5. `UnifiedDecoder::Forward`: Checks `enable_dflash_` and `dflash_draft_model_`

### global_token_num for Phase Detection

Use `global_token_num` to distinguish prefill vs decode:
- `global_token_num > 1`: Prefill mode (process prompt)
- `global_token_num == 1`: Decode mode (generate tokens)

### DFlash Statistics Flow

Statistics flow from C++ to Python API:
1. `Generation::GetDFlashStats` - tracks in `impl_->dflash_total_*_`
2. `LanguageModel::GetDFlashStats` - delegates to `impl->generation_->GetDFlashStats`
3. `Engine::GetDFlashStats` - delegates to `impl_->model_.GetDFlashStats`
4. `TurboMind::GetDFlashStats` - Python binding exposes via `bind.cpp`
5. Python wrapper `TurboMind.get_dflash_stats()` - user-facing API with calculated fields

Access via `model_comm.get_dflash_stats(index)` on the C++ TurboMind instance.

---

## 2026-05-14 - US-001: DFlash Decode Mode Execution

**Status**: ✅ Already implemented

All acceptance criteria for US-001 are already implemented in the codebase:

1. ✅ `[DFlash] === DECODE MODE: Verifying draft tokens ===` (unified_decoder.cc:412)
2. ✅ `[DFlash] Found stored draft tokens` (unified_decoder.cc:418)
3. ✅ `[DFlash] VerifyDraft: accepted=X tokens` (unified_decoder.cc:430)
4. ✅ `[DFlash] Using X accepted tokens from DFlash` (generation.cc:311)
5. ✅ generation.cc 中的 `dflash_accepted_tokens` 检查逻辑 (generation.cc:295)

**Files Changed**: None (already implemented)
- `src/turbomind/models/llama/unified_decoder.cc` - DECODE/PREFILL mode handling
- `src/turbomind/generation/generation.cc` - Accepted tokens usage

**Learnings:**
- The `EnableDFlash` flow works correctly but was previously confused by warmup phase logs
- Use `global_token_num` to detect prefill vs decode (NOT the `phase` variable)
- Draft tokens are stored in `args["dflash_stored_draft_tokens"]` and verified on next decode
- After verification, accepted tokens are output via `args["dflash_accepted_tokens"]`

---

## 2026-05-14 - US-002: Add Complete DFlash Statistics

**Status**: ✅ Completed

### What was implemented

Added `get_dflash_stats()` method to `TurboMind` class in `lmdeploy/turbomind/turbomind.py`. This method:
1. Calls `model_comm.get_dflash_stats(index)` to get raw statistics from C++ layer
2. Calculates `accept_rate` = total_accepted_tokens / total_draft_tokens
3. Calculates `speedup_ratio` = estimated speedup relative to baseline (45 tok/s)
4. Returns dict with all required fields: total_draft_steps, total_draft_tokens, total_accepted_tokens, total_rejected_tokens, accept_rate, speedup_ratio

### Files changed

- `lmdeploy/turbomind/turbomind.py`: Added `get_dflash_stats()` method (~20 lines)

### C++ Implementation (already existed)

The C++ side was already fully implemented:
- `generation.h/cc`: `Generation::GetDFlashStats` method with `total_rejected_tokens` parameter
- `generation.cc`: `dflash_total_rejected_tokens_` counter incremented when sampling rejected tokens (line 321)
- `bind.cpp`: Python binding exposed as `get_dflash_stats(index)` returning dict with accept_rate calculated

### Acceptance Criteria Verification

| Criteria | Status |
|----------|--------|
| Python API `tm.get_dflash_stats(0)` returns dict | ✅ Implemented |
| Contains total_draft_steps | ✅ |
| Contains total_draft_tokens | ✅ |
| Contains total_accepted_tokens | ✅ |
| Contains total_rejected_tokens | ✅ |
| Contains accept_rate (0-1) | ✅ Calculated in wrapper |
| Contains speedup_ratio | ✅ Calculated in wrapper |
| C++ tracks dflash_total_rejected_tokens_ | ✅ Already existed |
| GetDFlashStats signature has rejected_tokens | ✅ Already existed |

**Learnings:**
- C++ statistics tracking already existed - no changes needed there
- Python binding in `bind.cpp` already exposed `get_dflash_stats` method
- Missing was only the Python wrapper method in `TurboMind` class for user-facing API
- `model_comm` is the C++ TurboMind instance (`_tm.TurboMind.create()`) accessed via pybind11
- Statistics flow: C++ `Generation` → `LanguageModel` → `Engine` → `TurboMind` → Python binding → User API

---

