# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **Module tree uses X-macros**: All module types (ModelWeight, DecoderLayerWeight, AttentionWeight, etc.) use `TM_MODULE_DECLARE` / `TM_MODULE_METHODS` X-macro pattern for declaring children and params. Children are `unique_ptr<T>`, params are `core::Tensor`. Key file: `src/turbomind/core/module.h`.
- **Builder pattern for module creation**: Python builders (TextModelBuilder, DecoderLayerBuilder, etc.) in `lmdeploy/turbomind/builders/` coordinate C++ module creation via `create_module(Config)`, then commit tensors via `add_*` methods.
- **ModelWeight root children**: `output` (LinearWeight), `norm` (NormWeight), `layers` (ModuleList<DecoderLayerWeight>), plus `tok_embeddings` param. Defined in `MODEL_WEIGHT_CHILDREN` / `MODEL_WEIGHT_PARAMS` macros.
- **ModuleList is the only container**: All repeated modules (layers, experts) go through `ModuleList`, which maintains both named and indexed access.

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
