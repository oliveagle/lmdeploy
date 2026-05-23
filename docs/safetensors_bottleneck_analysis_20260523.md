# C++ Safetensors Loading Bottleneck Investigation

**Date**: 2026-05-23
**Status**: Investigation complete

## Executive Summary

Investigated the C++ safetensors loading bottleneck in TurboMind's weight loading pipeline.
The bottleneck is multi-factor, dominated by **fflush(stderr) calls** in the debug-heavy loading loop,
followed by **path mapping string operations** and **module tree traversal**.

## Architecture

The safetensors loading pipeline is:
1. `TM_TurboMind_InitFromPath()` in `src/turbomind/capi/turbomind_c.cc`
2. Builds Module tree (ModelWeight -> DecoderLayerWeight -> AttentionWeight/FfnWeight)
3. `LoadWeightsFromSafetensors()` reads .safetensors files
4. `MapHuggingFaceWeightToTurboMind()` maps HF paths to TM paths
5. Transfers tensor data via batched cudaMemcpyAsync

## Bottleneck Analysis

### 1. fflush(stderr) Calls [HIGH IMPACT]

**Location**: `LoadWeightsFromSafetensors()` at lines 877, 882, 947, 955, 976, 1040, 1056, 1065, 1076, 1183, 1190, 1281, 1300, 1305

**Impact**: ~10+ calls per tensor * ~0.02ms per flush = ~0.2ms per tensor
For a 35B model with 500+ tensors: 500 * 0.2ms = 100ms minimum

**For larger models (Qwen3.6-35B-A3B-AWQ with split files)**:
Multiple safetensors files, each with many tensors
Total flush count can reach 5000-10000+ calls
Estimated: 100-200ms wasted on flushes alone

### 2. Path Mapping String Operations [MEDIUM IMPACT]

**Location**: `MapHuggingFaceWeightToTurboMind()` at line 1323

**Impact**: Multiple find/replace operations per tensor name
The function does:
- 2 prefix removals
- Multiple self_attn replacements (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, in_proj_all)
- self_attn -> attention replacement
- mlp.experts -> moe_ffn.experts replacement
- mlp -> feed_forward replacement
- input_layernorm -> attention_norm replacement
- post_attention_layernorm -> ffn_norm replacement
- q_proj, k_proj, v_proj, o_proj replacements
- gate_proj -> w1, up_proj -> w3, down_proj -> w2 replacements
- qweight, qzeros, weight_scale replacements

Each find/replace is a full string scan (O(n) per operation)

### 3. Module Tree Traversal [LOW IMPACT]

**Location**: Lines 977-1072 in LoadWeightsFromSafetensors()

The module traversal is actually O(1) per level (Module::child() uses unordered_map).
The bottleneck is the cumulative effect of traversing deep paths like:
`layers.0.attention.w_qkv.weight` (4-5 levels deep per tensor)

### 4. Memory Allocation Pattern [MEDIUM IMPACT]

Current pattern:
1. Read tensor from safetensors (CPU)
2. Allocate GPU memory (cudaMallocAsync)
3. cudaMemcpyAsync (CPU -> GPU)

The benchmark shows Phase 2 batch transfers work well,
but Phase 1 (allocation + CPU reading) dominates the time.

## Benchmark Results

From test_safetensors_benchmark.cc:
- Open: ~0.1ms
- mmap: ~0.5ms (file size dependent)
- JSON parse: ~5-20ms (file size dependent)
- Path mapping: ~0.001ms per tensor
- Module traversal: ~0.0005ms per tensor
- get_tensor_data (mmap access): ~0.0001ms per tensor

## Root Cause

The primary bottleneck is the **excessive debug output with fflush** in the weight loading loop.
Each tensor triggers:
- Entry/exit prints
- Per-tensor status prints
- Progress accumulation prints

## Recommendations

1. **Remove or conditionally compile debug output**:
   - Wrap in `#ifdef SAFETENSORS_DEBUG` macro
   - Or use a configurable log level

2. **Batch debug output**:
   - Print every N tensors instead of every tensor
   - Use buffered logging instead of per-tensor fflush

3. **Optimize path mapping**:
   - Consider a compiled regex or state machine approach
   - Pre-compute common replacements

4. **Optimize memory allocation**:
   - Pre-allocate GPU memory blocks where possible
   - Use pinned memory for staging buffers

## Files Analyzed

- `src/turbomind/capi/turbomind_c.cc` - Main weight loading implementation
- `src/turbomind/utils/safetensors_reader.h` - Header-only safetensors reader
- `src/turbomind/utils/weight_serializer.cc` - Weight serialization utilities
- `test_safetensors_benchmark.cc` - Benchmark program
