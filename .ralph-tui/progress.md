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

