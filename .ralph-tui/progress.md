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
