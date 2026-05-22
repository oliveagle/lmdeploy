# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **TurboMind `data_type` semantics**: `EngineConfig.data_type` is the activation dtype, NOT weight dtype. AWQ weights are 4-bit (`kUint4`), but computations use FP16/BF16. The check at `turbomind.cc:152` requires `data_type == kHalf || kBfloat16`. `kHalf` is an alias for `kFloat16` in `data_type.h:80`. C API maps `TM_DATATYPE_FP16` → `kFloat16` via `FromCDataType()`.

---

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **TurboMind `data_type` semantics**: `EngineConfig.data_type` is the activation dtype, NOT weight dtype. AWQ weights are 4-bit (`kUint4`), but computations use FP16/BF16. The check at `turbomind.cc:152` requires `data_type == kHalf || kBfloat16`. `kHalf` is an alias for `kFloat16` in `data_type.h:80`. C API maps `TM_DATATYPE_FP16` → `kFloat16` via `FromCDataType()`.
- **VLM 模型 C++ 引擎加载限制**: Qwen3.5-9B-AWQ 是 VLM（视觉语言模型），包含 `vision_config` 和 linear attention (Mamba SSM) 层。C++ 引擎权重加载代码无法正确处理 `linear_attn.*` 权重（conv1d, dt_bias, in_proj_a/b/qkv/z, out_proj 等），全部被 SKIP_PARAM/SKIP_MODULE。QKV 融合在 layers 19/23/27/31 失败。`modules_to_not_convert` 在 config.json 中标记了 "visual", "linear_attn", "self_attn", "model.layers.0.", "mtp"，但 C++ 侧不读这个字段。需要支持 linear attention 层级的权重映射。

---

## [2026-05-22] - lmdeploy-s8l

### C++ 引擎端到端推理测试

**测试**: `./target/debug/examples/e2e_test /mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___5-9B-AWQ`

**结果**: 测试失败 - 权重加载不完整

**详细信息**:
- 加载了 24 个张量 (1+11+12+0+0)，大量张量被 SKIP_PARAM/SKIP_MODULE
- QKV 融合在 layers 19/23/27/31 失败: `Cannot find w_qkv.weight param`
- 最终错误: `vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)`

**根因分析**:
- Qwen3.5-9B-AWQ 是 VLM，包含 vision encoder + linear attention (Mamba SSM) 层
- C++ 引擎不支持 linear attention 层的权重映射 (`linear_attn.*` tensors)
- vision encoder 的 `model.visual.*` tensors 全部被 SKIP_MODULE
- `modules_to_not_convert` 字段标记了不应转换的模块，但 C++ 侧不读这个字段

**下一步**: 需要先完成 AWQ 权重加载支持（lmdeploy-109），再进行端到端测试

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

## [2026-05-22] - lmdeploy-h2w

### 修复 LinearWeight::param() 返回空的问题

**分析结果**: 代码已正确实现，无需修复

**详细说明**:
- `TM_MODULE_METHODS` 宏展开正确，生成正确的 `param()` 实现
- 符号 `turbomind::LinearWeight::param()` 正确导出到 `libturbomind_c.so`
- vtable 正确指向 LinearWeight 的实现，不是基类的空实现
- 汇编代码确认：使用 `strcmp` 比较字符串 "weight", "bias", "scales", "zeros"
- 参数名字符串匹配逻辑正确

**结论**: 如果运行时 `param("weight")` 仍返回空，问题可能在:
1. 调用 `param()` 的对象不是 LinearWeight 实例
2. 模块树中路径解析错误（parts.back() 不是 "weight"）
3. 权重文件路径映射错误，导致 param 分配未执行

---
