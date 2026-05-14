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

