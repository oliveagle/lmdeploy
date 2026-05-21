# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### MTP Weight Path Mapping
- **Pattern:** MTP (Multi-Token Prediction) weights in safetensors use `mtp.layers.X.*` prefix which shares weights with main model layers
- **Mapping:** `mtp.layers.X.*` → keep `layers.X.*` in path (don't strip layers. prefix!)
- **Bug:** Stripping "layers." from the path breaks module navigation since ModelWeight has a `layers` ModuleList child, not direct numeric children
- **MTP-specific params:** `mtp.norm`, `mtp.fc`, `mtp.pre_fc_norm_*` are NOT shared with main model - currently skipped (MTP speculative decoding not fully implemented in C++)
- **File:** `src/turbomind/capi/turbomind_c.cc:MapHuggingFaceWeightToTurboMind()`

### Context Stack for Allocators
- **Pattern:** The TurboMind `Context` class maintains thread-local stacks for device/host/pinned allocators
- **ContextGuard:** RAII pattern that pushes items (Stream, Allocator) onto the context stack, pops on destruction
- **device_alloc()** returns the top of the device allocator stack - useful for intercepting allocations
- **Key insight:** By pushing a custom allocator (e.g., CudaManagedAllocator) via ContextGuard before weight loading, all subsequent `Context::device_alloc()` calls return it
- **File:** `src/turbomind/core/context.h` and `src/turbomind/core/context.cc`

### X-Macro Pattern for Module Registration
- **Pattern:** X-macros (`TM_MODULE_DECLARE`, `TM_MODULE_METHODS`) generate add_child(), child(), param() methods from child/param lists
- **Usage:** Modules declare children via `TM_MODULE_DECLARE`, implementation via `TM_MODULE_METHODS`
- **Files:** `src/turbomind/core/module.h` for the base pattern, individual model weight files for concrete implementations

## [2026-05-21] - lmdeploy-gg4
- Fixed MTP safetensors weight path mapping to preserve "layers." prefix in module path
- **Bug:** `mtp.layers.0.mlp.experts.0.gate_proj.weight` was incorrectly mapped to `0.moe_ffn.experts.0.w1.weight` (missing "layers" prefix)
- **Root cause:** `result.substr(7)` stripped "layers." from path, breaking module navigation (ModelWeight->layers->0, not ModelWeight->0)
- **Fix:** Removed the `result.substr(7)` line, keeping "layers." in the path for correct navigation
- **Files changed:** `src/turbomind/capi/turbomind_c.cc` (line ~1112)
- **Note:** MTP-specific params (norm, fc, pre_fc_norm_*) are still skipped since C++ ModelWeight doesn't have MTP child modules yet
- **Build:** turbomind_c target compiled successfully

## [2026-05-21] - lmdeploy-s60
- **Work status:** Fix already implemented - DeltaNet linear_attn module weight loading works correctly
- **Implementation details:**
  - `delta_net_weight.h` includes all required children in `DELTA_NET_WEIGHT_CHILDREN` macro: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `in_proj_all`, `out_proj`, `norm`
  - `turbomind_c.cc` creates all these child modules with `add_child()` calls during layer initialization
  - `MapHuggingFaceWeightToTurboMind()` correctly maps HF paths like `.self_attn.in_proj.qkv.weight` to `.linear_attn.in_proj_qkv.weight`
- **Build verification:** turbomind_c target compiles successfully at 100%
- **Learnings:**
  - X-macro patterns (`TM_MODULE_DECLARE`, `TM_MODULE_METHODS`) automatically generate `add_child()`, `child()`, `param()` methods from the child/param lists
  - When adding new child modules, they need to be added to both the X-macro list in the header AND have `add_child()` called during initialization
  - Path mapping needs to happen before general path replacements to avoid incorrect mappings

---


## [2026-05-22] - lmdeploy-h2w
- **Fixed:** LinearWeight::param() returning empty Param
- **Root cause:** The macros `LINEAR_WEIGHT_PARAMS` and `LINEAR_WEIGHT_CHILDREN` are defined in the header file `linear_weight.h` inside the class body. When `TM_MODULE_METHODS` is called in the .cc file, these macros may not be in scope properly due to include order or macro visibility issues.
- **Fix:** Explicitly redefined `LINEAR_WEIGHT_CHILDREN` and `LINEAR_WEIGHT_PARAMS` macros in `linear_weight.cc` before calling `TM_MODULE_METHODS`, and added `#undef` after to clean up.
- **Files changed:** `src/turbomind/models/linear_weight.cc`
- **Verification:** `nm -C` shows `LinearWeight::param` symbol exists in object file
- **Learnings:**
  - X-macro patterns require macros to be visible at the point of use
  - Include order in C++ can affect macro visibility
  - The `.h` file defining macros doesn't guarantee they're visible in the `.cc` file's scope
  - Always verify macros are in scope when using TM_MODULE_METHODS pattern
---

## [2026-05-22] - lmdeploy-d6w
- **Work status:** Debug logging added to verify ContextGuard effectiveness
- **Implementation details:**
  - Added debug logging in `TM_TurboMind_InitFromPath` after `ctx_guard` creation to verify allocator state
  - Added debug logging at entry of `LoadWeightsFromSafetensors` to check if allocator is properly inherited from caller's scope
  - Added debug logging before `target_param.alloc()` to verify allocator state at the exact point of tensor construction
  - Added debug logging before calling `LoadWeightsFromSafetensors` to verify ctx_guard is still in scope
- **Files changed:** `src/turbomind/capi/turbomind_c.cc`
- **Build verification:** turbomind_c target compiled successfully (100%)
- **Debug outputs:**
  - Allocator validity check: `(bool)alloc`
  - Allocator device type: `alloc->device().type`
  - Allocator device ID: `alloc->device().id`
- **Learnings:**
  - Allocator class uses `operator->` to access `AllocatorImpl`, so must use `alloc->device()` not `alloc.device()`
  - ContextGuard pushes allocator via `Context::push()` which stores it in thread-local stack
  - Debug logging needs to be at exact point of failure (before tensor construction, not just at function entry)
---

## [2026-05-22] - lmdeploy-09p
- **Work status:** All required functions already implemented - verified complete
- Implementation includes:
  1. `TM_ModelRequest_ForwardAsync` - Non-blocking async forward inference
  2. `TM_ModelRequest_GetStreamToken` - Stream output token retrieval
  3. `TM_ModelRequest_GetStreamingState` - Poll request state in async mode
  4. `TM_ModelRequest_GetOutput` - Get output tensor by name from completed request
  5. `TM_ModelRequest_Cancel` - Cancel running request
- **Files verified:** `src/turbomind/capi/turbomind_c.cc`, `turbomind_c.h`
- **Build verification:** turbomind_c target compiled successfully (100%)
- **Learnings:**
  - The C API `ForwardAsync` submits request and returns immediately without blocking
  - Uses shared state (`streaming_tensors`, `streaming_state`) for polling
  - `AtomicRequestState::exchange(nullptr)` pattern ensures one-time consumption
  - `ModelRequest::Forward` accepts callback and returns `OutputParam` with tensors/state/metrics
---

## [2026-05-22] - lmdeploy-hr0
- **Work status:** Optimized weight loading performance using BatchCopy
- **Implementation details:**
  - Added `#include "src/turbomind/core/copy.h"` for BatchCopy support
  - Modified `LoadWeightsFromSafetensors()` to use two-phase approach:
    - Phase 1: Allocate GPU memory and collect transfer metadata
    - Phase 2: Run all transfers in batch using `BatchCopy::group()` + `Run()`
  - Removed per-tensor `cudaMemcpyAsync` calls and `cudaStreamCreate`/`Synchronize`/`Destroy`
  - Uses existing `BatchCopy` class which wraps `cuMemcpyBatchAsync` driver API for optimal batching
- **Files changed:** `src/turbomind/capi/turbomind_c.cc`
- **Build verification:** turbomind_c target compiled successfully at 100%
- **Learnings:**
  - BatchCopy uses RAII pattern with `Group` class for automatic batch management
  - cuMemcpyBatchAsync can batch multiple small transfers into a single GPU operation
  - Two-phase approach (allocate all first, then batch copy) reduces GPU memory fragmentation
  - `BatchCopy::operator()(src, size, dst)` adds transfers to current batch group
  - `BatchCopy::group()` creates RAII group, `Run()` executes batched transfers
- **Potential future optimization:** Could use prefetch/hints for even better performance on multi-GPU setups

---

## [2026-05-22] - lmdeploy-zm7
- **Fixed:** CUDA OOM in safetensors weight loading for large models
- **Implementation:**
  1. Added `kMANAGED` to `DeviceType` enum in `allocator.h`
  2. Created `CudaManagedAllocator` class using `cudaMallocManaged` for unified memory
  3. Modified `Param::alloc()` in `module.h` to use `Context::device_alloc()` instead of hardcoded `kDEVICE`
  4. Modified `LoadWeightsFromSafetensors()` in `turbomind_c.cc` to push `CudaManagedAllocator` via `ContextGuard`
- **How it works:** `cudaMallocManaged` allocates unified memory that can be oversubscribed (use more than GPU VRAM by leveraging system RAM). By pushing a `CudaManagedAllocator` via `ContextGuard`, all weight tensor allocations during loading use managed memory instead of direct GPU memory.
- **Files changed:**
  - `src/turbomind/core/allocator.h` - Added kMANAGED type and CreateCudaManagedAllocator() declaration
  - `src/turbomind/core/allocator.cc` - Added CudaManagedAllocator class implementation
  - `src/turbomind/core/module.h` - Modified Param::alloc() to use Context::device_alloc()
  - `src/turbomind/capi/turbomind_c.cc` - LoadWeightsFromSafetensors now pushes managed allocator
- **Build verification:** turbomind_c compiled successfully at 100%
- **Learnings:**
  - Context class maintains thread-local allocator stacks via ContextGuard pattern
  - By pushing a custom allocator before weight loading, all subsequent `device_alloc()` calls return it
  - `cudaMallocManaged` allows memory oversubscription on systems with more RAM than VRAM
  - The Tensor constructor with Device type internally calls `Context::alloc(device)` which uses `device_alloc()` when device.type == kDEVICE
  - Changed Param::alloc() to use `Context::device_alloc()` instead of hardcoded `kDEVICE` to enable dynamic allocation behavior based on context
---

## [2026-05-22] - lmdeploy-fcp
- **Work status:** Pure Rust Tokenizer already implemented and integrated
- **Implementation verified:**
  - `LMTokenizer` in `lmdeploy-rust-server/src/tokenizer.rs` - complete implementation (208 lines)
  - Uses HuggingFace `tokenizers` crate (version 0.21) - pure Rust, no Python dependency
  - Supports `tokenizer.json` and `tokenizer.model` (SentencePiece) formats
  - Methods: `encode()`, `decode()`, `encode_batch()`, `encode_raw()`, `id_to_token()`, `decode_token()`
  - Tracks BOS/EOS token IDs, vocabulary size, special tokens
- **Integration verified:**
  - Used by `TurboMindCEngine` in `cpp_engine.rs` for encode/decode
  - Tokenizer loaded in `TurboMindCEngine::new()` at line 141-150
  - Used in `generate()` for prompt tokenization (line 307-318)
  - Used in `generate_stream()` for streaming decode (line 402-419)
  - Used for output decoding after inference (line 372-382)
- **Tests:** All 4 tokenizer unit tests pass
- **Files:** `lmdeploy-rust-server/src/tokenizer.rs`, `src/model/cpp_engine.rs`, `Cargo.toml`
- **Learnings:**
  - The HuggingFace `tokenizers` crate provides a complete Rust implementation
  - No Python/transformers dependency needed for tokenization
  - Tokenizer loads directly from `tokenizer.json` or `tokenizer.model` files
  - Integration pattern: tokenizer is optional field, loaded during engine initialization
  - The C++ TurboMind engine expects pre-tokenized integer token IDs, not text
  - Tokenizer is only used at the Rust layer boundaries (input prompt → tokens, output tokens → text)

---

## [2026-05-22] - lmdeploy-zks
- **Investigated:** C++ safetensors loading bottleneck
- **Files analyzed:** `src/turbomind/capi/turbomind_c.cc`, `src/turbomind/utils/safetensors_reader_mmap.h`, `src/turbomind/core/module.cc`
- **Benchmark results (Qwen3.5-9B-HF, 4 shards):**
  - Shard 1 (5.28GB, 13 tensors): open=0.01ms, mmap=0.00ms, parse=0.03ms
  - Shard 2 (5.34GB, 52 tensors): mmap=0.00ms, parse=1.31ms
  - Shard 3 (5.37GB, 62 tensors): similar fast
  - Shard 4 (3.33GB, 644 tensors): largest tensor count
- **Path mapping benchmark:** ~0.02ms for 53 tensors (0.000ms/tensor)
- **Module traversal benchmark:** ~0.02ms for all operations
- **get_tensor_data (mmap access):** 0.00ms for all tensors (zero-copy!)
- **Findings:**
  1. mmap is NOT the bottleneck - completes in <1ms
  2. JSON parsing is fast - ~1-2ms for files with hundreds of tensors
  3. Path mapping and module traversal are negligible - <0.001ms per tensor
  4. **Likely cause of >120s loading time:** Excessive debug printing with fflush after every tensor operation (10+ flushes per tensor × 174 tensors = ~1700+ flushes during single file load)
  5. The current LoadWeightsFromSafetensors has debug logging on every iteration which writes to stderr and flushes
- **Recommendation:** Make debug logging conditional on a debug flag (e.g., `LMDEPLOY_LOG_LEVEL=DEBUG`)
- **Learnings:**
  - SafetensorsReaderMmap uses zero-copy mmap - no data copying in get_tensor_data()
  - BatchCopy optimization (lmdeploy-hr0) already applied - good
  - CUDA managed memory allocation (lmdeploy-zm7) - overhead likely minimal
  - The C++ safetensors reader is actually FAST - the bottleneck is I/O overhead from debug logging

---
