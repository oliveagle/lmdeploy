# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- **E2E test structure**: Multi-layered test structure: `tests/` (CPU-only unit/integration tests), `examples/` (GPU e2e tests), `bin/` (quick binary tests), Python integration tests
- **Test layering principle**: CPU-only tests in `tests/` for quick feedback; GPU tests in `examples/` and `bin/` require model paths
- **C API integration pattern**: All C++ engine tests follow: `EngineConfig` → `TurboMind` → `ModelRequest` → `TensorMap` → inference
- **Phase-based test design**: All e2e tests use phase-based logging ([Phase 1], [Phase 2], etc.)
- **Module tree uses X-macros**: All module types (ModelWeight, DecoderLayerWeight, AttentionWeight, etc.) use `TM_MODULE_DECLARE` / `TM_MODULE_METHODS` X-macro pattern for declaring children and params. Children are `unique_ptr<T>`, params are `core::Tensor`. Key file: `src/turbomind/core/module.h`.
- **Builder pattern for module creation**: Python builders (TextModelBuilder, DecoderLayerBuilder, etc.) in `lmdeploy/turbomind/builders/` coordinate C++ module creation via `create_module(Config)`, then commit tensors via `add_*` methods.
- **ModelWeight root children**: `output` (LinearWeight), `norm` (NormWeight), `layers` (ModuleList<DecoderLayerWeight>), plus `tok_embeddings` param. Defined in `MODEL_WEIGHT_CHILDREN` / `MODEL_WEIGHT_PARAMS` macros.
- **ModuleList is the only container**: All repeated modules (layers, experts) go through `ModuleList`, which maintains both named and indexed access.
- **AttentionWeight uses fused QKV (w_qkv)**: Python `AttentionBuilder.add_qkv_proj()` fuses separate q/k/v weights into a single `w_qkv` child. C++ must create `w_qkv` instead of separate `q_proj`/`k_proj`/`v_proj`. The fused shape is `[q_out + k_out + v_out, hidden]`. HF stores separate weights with `[hidden, out]` shape that need transposition during fusion.
- **byte_size function**: Use `turbomind::byte_size(dtype)` (not `DataTypeSize`) to get element byte size. Defined in `src/turbomind/core/data_type.h`.

---

## 2026-05-22 - lmdeploy-zk3
- Verified complete e2e test infrastructure for pure Rust+C++ inference
- **Files checked**:
  - `tests/e2e_test.rs`: Rust unit/integration tests (CPU-only, no GPU)
  - `examples/e2e_test.rs`: Full GPU e2e test with C++ engine
  - `tests/e2e_integration.py`: Python integration tests (tokenizer, config)
  - `bin/cpp_engine_test.rs`: Quick C++ engine test binary
- **Test coverage**:
  - Tokenizer loading and encode/decode roundtrip
  - Model config parsing (AWQ detection)
  - Engine type selection (PythonBridge vs PureCpp)
  - ModelState transitions
  - Full C++ engine initialization (TurboMind)
- **Verification**: All tests compile with `cargo build`
- **Learnings**:
  - Test files follow phase-based logging pattern for easy debugging
  - CPU-only tests are prioritized in `tests/` directory for quick feedback
  - GPU tests live in `examples/` and `bin/` which require model paths

---

## 2026-05-22 - lmdeploy-5wx
- Analyzed ModelWeight complete module tree structure
- Documented full hierarchy: ModelWeight → layers[] → DecoderLayerWeight → {attention_norm, attention, ffn_norm, feed_forward/moe_ffn}
- Identified 5 missing steps in C API: module tree creation, tensor slot allocation, weight data loading, prepare(), verify()
- Created model_weight_tree.md with structure diagram and reference file list
- **Files changed**: `.ralph-tui/model_weight_tree.md` (created)
- **Learnings:**
  - Module tree uses X-macros (TM_MODULE_DECLARE / TM_MODULE_METHODS) - all modules follow same pattern
  - Python builders coordinate C++ module creation - C API needs to replicate this flow
  - ModuleList is the single container for all repeated modules (layers, experts)
  - ModelWeight derives data_type, hidden_units, vocab_size from children in prepare()
---

## 2026-05-22 - lmdeploy-85l
- Analyzed C++ engine AWQ loading failure: `TM][FATAL] Check failed: data_type_ == kBfloat16 || data_type_ == kHalf`
- **Root cause identified**: `EngineConfig::data_type` field had no default value, zero-initializing to `kNull` (0) instead of `kHalf` (66826)
- **Key findings**:
  - `ENGINE_FIELDS(X)` macro in `engine_config.h` line 15: `X(DataType, data_type)` - no default specified
  - `TM_MEMBER(Type, name, ...)` expands to `Type name{__VA_ARGS__}` → `Type name{}` when empty → zero initialization
  - `DataType::kNull = 0` is neither `kBfloat16` (67591) nor `kHalf` (66826), causing check failure
  - C API enum `TM_DATATYPE_FP16` (10) correctly converts to C++ `kFloat16` (66826) via `FromCDataType`
  - `kHalf = kFloat16` by definition, so conversion was correct but default was missing
- **Files changed**:
  - `src/turbomind/engine/engine_config.h`: Added `kHalf` default to `data_type` field
  - `src/turbomind/capi/turbomind_c.cc`: Removed redundant explicit assignment (now uses default)
- **Learnings**:
  - X-macro fields without defaults become zero-initialized, not undefined
  - C++ `DataType` uses encoded values (sign<<16 | exponent<<8 | mantissa), not sequential integers
  - C API `TM_DataType` uses sequential integers (0, 1, 2, ...) - must convert via `FromCDataType`
  - `kHalf = kFloat16` alias means both names refer to same encoded value (66826)
  - AWQ weights use `kUint4` storage but compute in `kHalf` - data_type is activation dtype, not weight dtype
---

## 2026-05-22 - lmdeploy-4rt
- Fixed weight loading path mapping: HF q_proj/k_proj/v_proj → TM fused w_qkv
- **Files changed**:
  - `src/turbomind/capi/turbomind_c.cc`:
    - Changed `InitFromPath` to create `w_qkv` (fused QKV) instead of separate `q_proj`, `k_proj`, `v_proj`
    - Updated `MapHuggingFaceWeightToTurboMind` to map `.q_proj.`, `.k_proj.`, `.v_proj.` to `.w_qkv.`
    - Added QKV fusion logic in `LoadWeightsFromSafetensors` to accumulate Q/K/V tensors and fuse them
    - Fixed data type references (use `turbomind::DataType` and `turbomind::byte_size`)
- **Learnings**:
  - Python `AttentionBuilder.add_qkv_proj()` fuses separate q/k/v into single `w_qkv` child
  - `AttentionWeight` class defines `w_qkv` (fused), not separate `k_proj`/`v_proj` children
  - Fused shape: `[q_out + k_out + v_out, hidden]`, where HF stores `[hidden, out]` each
  - QKV fusion requires transpose during concatenation (HF column-major → TM row-major)
  - Must accumulate all 3 tensors before allocating fused GPU memory

---
