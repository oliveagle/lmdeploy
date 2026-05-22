# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **TurboMind `data_type` semantics**: `EngineConfig.data_type` is the activation dtype, NOT weight dtype. AWQ weights are 4-bit (`kUint4`), but computations use FP16/BF16. The check at `turbomind.cc:152` requires `data_type == kHalf || kBfloat16`. `kHalf` is an alias for `kFloat16` in `data_type.h:80`. C API maps `TM_DATATYPE_FP16` → `kFloat16` via `FromCDataType()`.

---

## [2026-05-22] - lmdeploy-85l

### Analysis: C++ Engine AWQ data_type Check Failure

**Root Cause**: The check at `turbomind.cc:152` (`data_type_ == kBfloat16 || data_type_ == kHalf`) is correct. The `data_type` field represents activation dtype, not weight dtype:

- For AWQ models: activation = `kHalf` (FP16), weights = `kUint4` (4-bit quantized)
- `kHalf` is an alias for `kFloat16` (data_type.h:80)
- The check requires activation to be either FP16 or BF16

**Code Flow**:
1. Rust: `set_data_type(TM_DATATYPE_FP16)` calls C API `TM_EngineConfig_SetDataType`
2. C API: `FromCDataType(TM_DATATYPE_FP16)` → `kFloat16`
3. C++: `data_type_ = config.data_type` → `kFloat16`
4. Check: `kFloat16 == kHalf` → TRUE (they're the same enum)

**Key Files**:
- `src/turbomind/core/data_type.h` - DataType enum, `kHalf = kFloat16`
- `src/turbomind/engine/engine_config.h:15` - `data_type` default = `kHalf`
- `src/turbomind/turbomind.cc:152` - Activation dtype check
- `src/turbomind/capi/turbomind_c.cc:176-178` - C API data_type setter

**Resolution**: The Rust code correctly sets `data_type = TM_DATATYPE_FP16`. The failure could be from the `config` being moved before TurboMind reads it, or from enum value mismatch between Rust and C++ sides.

---
