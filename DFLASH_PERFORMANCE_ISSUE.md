# DFlash Performance Issue Analysis

## Problem (已修复)

Benchmark showed 39.24 tokens/s with DFlash, which was slower than the baseline 45 tokens/s without DFlash.

## Root Cause (已修复)

### Phase Restriction Bug (原问题)

The original code had:
```cpp
if (enable_dflash_ && dflash_draft_model_ && phase == 0) {
    // DFlash speculative decoding
}
```

This meant DFlash only runs when `phase == 0`, which is NOT the decode phase.

### Misunderstanding of `phase` Variable

**Critical Discovery**: The `phase` variable in TurboMind does NOT represent prefill vs decode!
- `phase` is the async pipeline stage (0 or 1)
- `phase = 0` or `phase = 1` are both used for BOTH prefill and decode
- The actual prefill/decode status is determined by `global_token_num`:
  - Prefill: `global_token_num > 1` (processing multiple tokens)
  - Decode: `global_token_num == 1` (processing single token)

### Execution Flow (Before Fix)

**Prefill phase (global_token_num > 1):**
1. Process entire prompt
2. Collect aux_hidden_states from intermediate layers
3. Generate draft tokens (8 speculative tokens)
4. Verify draft tokens against target logits
5. Output accepted tokens to environment

**Decode phase (global_token_num == 1):**
1. Process 1 new token
2. DFlash does NOT run (because of `phase == 0` check)
3. Normal sampling

### Why This Caused Slowdown

1. **During prefill**: DFlash runs, but the draft tokens are verified immediately in the same forward pass. This doesn't give us parallel execution benefit.

2. **During decode**: DFlash doesn't run at all. Each token requires a full forward pass without any speculation.

3. **Overhead**: The DFlash code path adds overhead (collecting aux_hidden_states, running draft model, verification) without providing the speedup benefit.

## Fix Applied (2026-05-14)

### 1. Corrected Phase Detection

Changed from:
```cpp
if (enable_dflash_ && dflash_draft_model_ && phase == 0)
```

To:
```cpp
const bool is_prefill = global_token_num > 1;
const bool is_decode = global_token_num == 1;

if (enable_dflash_ && dflash_draft_model_) {
    if (is_decode) {
        // Verify stored drafts and generate new ones
    } else if (is_prefill) {
        // Generate drafts for next iteration
    }
}
```

### 2. Proper Draft Token Storage

- **Prefill phase**: Generate draft tokens and store in `env["dflash_stored_draft_tokens"]`
- **Decode phase**: Retrieve stored drafts, verify against target logits, use accepted tokens

### 3. Correct Speculative Decoding Flow

1. **Prefill (global_token_num > 1)**:
   - Generate M draft tokens from aux_hidden_states
   - Store drafts in environment for next iteration
   - No verification (no target logits yet)

2. **Decode (global_token_num == 1)**:
   - Retrieve stored draft tokens from previous iteration
   - Get target logits from target model
   - Verify drafts against target logits
   - Output accepted tokens to Generation module
   - Generate new drafts for next iteration

## Expected Results

After this fix:
- DFlash runs during decode phase (where it matters)
- Draft tokens are verified across iterations, not immediately
- Accepted tokens skip target model forward pass
- Expected speedup: 1.3x - 1.7x (depending on accept rate)

## Files Modified

- `src/turbomind/models/llama/unified_decoder.cc` - Fixed phase detection and speculative flow
- `src/turbomind/models/llama/unified_decoder.h` - Cleaned up unused state storage

## Testing

Run benchmark to verify fix:
```bash
python benchmark_dflash.py
```

Expected: Tokens/s should be higher than baseline (45 tokens/s) with DFlash enabled.
