# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **TurboMind Module System**: All weight modules use X-macros (`TM_MODULE_DECLARE`) to declare children and params. Python uses `Builder` pattern to stage children/tensors, then `build()` creates C++ handles via `_tm.create_module(config)` and attaches them with `add_child_raw(name, child)`. Tensor data is allocated with `param.alloc(shape, dtype)` then copied with `dst.copy_from(tensor)`.
- **ModelWeight Tree**: `ModelRoot -> text_model (ModelWeight) -> tok_embeddings(tensor), output(LinearWeight), norm(NormWeight), layers(ModuleList) -> DecoderLayerWeight[40] -> attention(AttentionWeight), feed_forward(FfnWeight), attention_norm/ffn_norm(NormWeight)`. DeltaNetWeight and MoeWeight are conditional children.
- **C API gap**: `InitFromPath()` creates empty `ModelWeight` without building the child module tree or loading weight data. Python `ModelLoader.export()` calls `model.model(Prefix(ckpt))` which builds the full tree via Builder pattern.
- **AttentionWeight Children**: Has 10 possible children (w_qkv, wo, q_proj, q_a_proj, q_b_proj, kv_a_proj, q_norm, k_norm, q_a_layernorm, kv_a_layernorm) and 1 param (sinks). Full attention models use q_proj/k_proj/v_proj/wo; MLA models use q_a_proj/q_b_proj/kv_a_proj/q_norm/k_norm/q_a_layernorm/kv_a_layernorm.
- **DeltaNetWeight params**: Has 3 params (conv1d, A_log, dt_bias) that are NOT allocated in C++ InitFromPath - these come from safetensors and need to be loaded like other weights.
- **ModuleList indexing**: ModuleList children are added with string indices ("0", "1", etc.), not numeric indices. Access via `layers_list->add_child(std::to_string(layer_idx), ...)` or `layers->child("0")`.
- **ModelRoot child naming**: ModelWeight is attached to ModelRoot as "text_model" child via `model_root->add_child("text_model", ...)`. Get back via `model_root->text_model_ptr()`.

- **Rust FFI Naming Convention**: FFI type enums use C-style `TM_DATATYPE_*` naming (not Rust camelCase) to match C++ symbols. Clippy warnings are expected but acceptable for C interop.
- **TurboMind Init Lifecycle**: C API `InitFromPath()` performs full initialization: `CreateContext → CreateRoot → Build ModelWeight tree → Load safetensors → ProcessWeights (GPU transfer) → CreateEngine`. Single call handles everything.
- **AWQ Detection**: Engine auto-detects AWQ quantization by reading `config.json` for `"quant_method": "awq"` pattern, then sets `quant_policy=4` automatically.

---

## 2026-05-22 - lmdeploy-m9v
- Verified Rust FFI bindings already fully implemented in `turbomind_c.rs` (1278 lines)
- All C API types wrapped in RAII Rust types: `EngineConfig`, `TurboMind`, `TensorMap`, `GenConfig`, `ModelRequest`, `SafetensorsHandle`
- `cpp_engine.rs` integrates FFI bindings for pure C++ inference (no Python dependency)
- Streaming generation support via `forward_async` + polling loop
- Code compiles cleanly, tests pass
- **Files changed**: N/A (verified existing implementation, no changes needed)
- **Learnings:**
  - FFI types intentionally use C-style naming (`TM_DATATYPE_*`) to match C++ ABI
  - `ModelRequest::forward()` accepts mutable `TensorMap` for output results
  - Streaming tokens read via `get_stream_token()` + `get_streaming_state()` polling
  - AWQ models auto-detected at load time from `config.json`
---
- Analyzed and verified the complete ModelWeight module tree structure implemented in C++ InitFromPath
- Verified: ModelWeight creates tok_embeddings param, norm child (NormWeight), output child (LinearWeight), layers ModuleList
- Verified: Each DecoderLayerWeight creates attention_norm, ffn_norm, attention (AttentionWeight with q_proj/k_proj/v_proj/wo), feed_forward (FfnWeight with w1/w2/w3)
- Verified: MoE models (Qwen3.6-35B-A3B) create moe_ffn with gate and experts ModuleList
- Verified: Linear attention models create linear_attn (DeltaNetWeight) with in_proj_qkv/z/a/b/all, out_proj, norm children
- Verified: Build succeeds without errors
- **Files changed**: N/A (verified existing implementation, no changes needed)
- **Learnings:**
  - AttentionWeight has 10 possible children but only uses q_proj/k_proj/v_proj/wo for standard full attention (Qwen3)
  - MLA (Multi-head Latent Attention) models use different children (q_a_proj, q_b_proj, kv_a_proj, q_norm, k_norm, etc.)
  - DeltaNetWeight has 3 params (conv1d, A_log, dt_bias) that need allocation if present in safetensors
  - Current implementation handles both MoE and linear attention models correctly
  - Module creation follows X-macro pattern: Module::create(config) -> unique_ptr -> add_child(name, move(ptr))
---
- Analyzed Python TurboMind ModelWeight construction and module tree structure
- Built complete documentation of ModelWeight hierarchy with all children modules and parameters
- Identified C API missing steps: module tree creation, tensor allocation, weight copying
- **Files changed**: Created `src/turbomind/models/MODULE_TREE_STRUCTURE.md`
- **Learnings:**
  - Module system uses X-macros (`TM_MODULE_DECLARE`) for declarative child/param registration
  - Python `Builder` pattern stages children/tensors into `_pending_children`/`_pending_tensors` dicts, then `build()` drains them to C++ via `create_module()` + `add_child_raw()` + `param.alloc()` + `copy_from()`
  - `DecoderLayerWeight` has 6 children: `attention`, `linear_attn` (DeltaNet, conditional), `feed_forward`, `moe_ffn` (conditional), `attention_norm`, `ffn_norm`
  - `AttentionWeight` has 10 possible children (MLA variants: q_proj, q_a_proj, q_b_proj, kv_a_proj, q_norm, k_norm, etc.) — actual children depend on model architecture
  - C API `InitFromPath()` creates an empty `ModelWeight` via `create_module()` but skips all child construction and weight loading steps
---
