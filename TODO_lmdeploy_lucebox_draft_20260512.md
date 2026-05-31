# DFlash + DTree Implementation Task List

## Current Status (2026-05-12)

### ✅ Completed
- DFlashDraftModel class implementation
- Non-causal attention kernel in dflash_kernels.cu
- Hidden states collection from 5 layers {1, 8, 16, 24, 31}
- TurboMind Python integration with speculative_config
- Basic loading and initialization path working

### ❌ Issues
- GenerateDraft/VerifyDraft not executing after hidden states collection
- Hidden size mismatch (draft: 5120 vs target: 4096)
- No actual speculative token generation/verification happening yet
- No DTree verification implemented yet

---

## Task Breakdown

### Phase 1: Fix Speculative Decoding Execution (High Priority)

**Task 1.1: Debug GenerateDraft execution**
- [ ] Add detailed trace logging at GenerateDraft entry point
- [ ] Check aux_hidden_states tensor shapes before GenerateDraft
- [ ] Verify DFlashDraftModel::SetDraftWeightPointer set correctly
- [ ] Check for CUDA sync errors after hidden state copies
- [ ] Validate cuBLAS handle initialization in DFlashDraftModel constructor

**Task 1.2: Fix hidden size mismatch**
- [ ] Investigate draft model config hidden size 5120 vs target 4096
- [ ] Check if draft weight loading uses wrong dimensions
- [ ] Verify DFlashDraftWeight init uses correct hidden size from target model
- [ ] Fix weight map loading in LoadDFlashWeightsQuantized

**Task 1.3: Enable full GenerateDraft execution**
- [ ] Add logging for each step in GenerateDraft
- [ ] Verify 5 hidden states are correctly shaped (layer-wise)
- [ ] Check for segfault/crash in the NoiseEmbed path
- [ ] Ensure attention kernel can run without KV cache

### Phase 2: Implement Verification Flow

**Task 2.1: VerifyDraft implementation**
- [ ] Verify target logits tensor comes from correct layer output
- [ ] Add step-by-step logging in VerifyDraft
- [ ] Check token probabilities calculation with softmax
- [ ] Verify acceptance rate is logged and calculated properly

**Task 2.2: Integration with Generation**
- [ ] Verify dflash_accepted_tokens is produced to args
- [ ] Check Generation code path picks up accepted tokens
- [ ] Ensure speculative decoding actually reduces decode steps

### Phase 3: Implement DDTree Verification (Key Speed Boost)

**Task 3.1: DDTree structure implementation**
- [ ] Design DDTree node structure (token + logits + children)
- [ ] Implement tree construction from draft tokens (budget=22)
- [ ] Implement verification tree traversal logic

**Task 3.2: Tree-based speculative decoding**
- [ ] Generate 8 draft tokens + tree expansion
- [ ] Verify all tree nodes with target logits
- [ ] Calculate max consecutive accepted tokens
- [ ] Return accepted tokens with highest probability path

### Phase 4: Performance Benchmarking

**Task 4.1: Baseline performance without DFlash**
- [ ] Run Qwen3.5-9B-AWQ baseline decode speed
- [ ] Measure single-user (chat) throughput
- [ ] Measure batch decoding throughput

**Task 4.2: DFlash performance measurement**
- [ ] Run DFlash with speculative decoding
- [ ] Measure acceptance rate (target 60-80%)
- [ ] Measure decode speed-up (target 1.7-2.0x)
- [ ] Verify tree-based verification gives 30% boost over linear verify

### Phase 5: Cleanup & Optimization

**Task 5.1: Code cleanup**
- [ ] Remove debugging printfs/logs
- [ ] Add proper comment documentation
- [ ] Fix code formatting
- [ ] Remove unused draft weights from memory after initialization

**Task 5.2: Performance optimizations**
- [ ] Optimize hidden state collection memory usage
- [ ] Verify cuBLAS calls are efficient
- [ ] Check if we can avoid unnecessary memory copies
- [ ] Optimize tree construction overhead

---

## Test Scenarios

**Test 1: Basic DFlash generation**
- Prompt: "Python is a"
- Check if draft tokens are generated
- Check if verification works correctly

**Test 2: Acceptance rate measurement**
- 100 different prompts (short/medium/long)
- Measure average accept rate
- Log distribution of accepted counts

**Test 3: Throughput benchmark**
- Chat mode (1 prompt at a time)
- Batch mode (10 parallel prompts)
- Measure tokens/sec throughput with/without DFlash

---

## Target Metrics (From lucebox)

| Metric | Target | Current |
|--------|--------|---------|
| Decode speedup | 1.7-2.0x | TBD |
| Acceptance rate | 60-80% | TBD |
| Draft tokens per step | 8 | Configured to 8 |
| Tree budget (DDTree) | 22 | Not implemented |

---

## Important Files

### Core Files
- `src/turbomind/models/llama/DFlashDraftModel.{h,cu}` - Draft model
- `src/turbomind/models/llama/DFlashDraftWeight.{h,cu}` - Draft weight
- `src/turbomind/models/llama/dflash_kernels.{h,cu}` - CUDA kernels
- `src/turbomind/models/llama/unified_decoder.{h,cc}` - Layer loop & hidden state collection

### Python Integration
- `lmdeploy/turbomind/turbomind.py` - TurboMind Python entry
- `lmdeploy/turbomind/model_executor.py` - Model execution
- `test_dflash_detailed.py` - Test script

### Lucebox Reference
- Check `../lucebox-hub/dflash/` directory structure
- Compare with LMDeploy implementation

---

## Notes

1. **KV Cache vs No KV Cache**: Lucebox has NO KV cache - check if LMDeploy's KV cache usage in draft model is causing issues
2. **Memory Layout**: Verify hidden states are in the format DFlashDraftModel expects
3. **DDTree**: Tree verification gives ~30% extra speed boost on top of linear speculative decoding
4. **Quant Policy**: Target model uses AWQ quant_policy=8, draft model uses quant_policy=0
