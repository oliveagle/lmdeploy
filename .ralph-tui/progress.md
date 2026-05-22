# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-22] - lmdeploy-mdk
- 删除 lmdeploy-rust-server 中的 Python Bridge 代码和依赖
- **Files deleted**:
  - `src/model/python_bridge.rs` - Python subprocess bridge 445行
  - `src/model/engine.rs` - Python bridge 集成 342行
  - `scripts/python_inference_bridge.py` - Python 推理桥接脚本
  - `scripts/convert_hf_to_turbomind.py` - 模型转换脚本
  - `inference_helper.py` - 推理辅助脚本
- **Files modified**:
  - `src/model/mod.rs` - 移除 python_bridge 模块，简化导出
  - `src/model/manager.rs` - 移除 `ModelEngine::PythonBridge` 枚举变体和所有相关代码
  - `src/model/cpp_engine.rs` - 移除 `EngineType::PythonBridge`，只保留 `PureCpp`
  - `src/config.rs` - 更新注释（engine_type 只支持 pure_cpp）
  - `src/server.rs` - 移除 `PythonBridge` 默认回退
  - `src/model/benchmark.rs` - 更新为使用 `TurboMindCEngine` 而非 `TurboMindEngine`
- **Learnings:**
  - 删除代码时注意 `model/mod.rs` 中的导出要同步更新
  - `ModelInfo` 结构在 `cpp_engine.rs` 中定义，需要导入后才能在 `manager.rs` 使用
  - `is_ready` 字段是 `AtomicBool`，调用时需要 `.load()`
  - 保留 `load_model` 方法（默认使用 PureCpp）以保持 API 兼容性
---

## [2026-05-22] - lmdeploy-109 (Completed)
- **Verified**: C++ AWQ weight loading already fully implemented
- Files verified:
  - `src/turbomind/capi/turbomind_c.cc` - `CreateAwqLinearConfig()`, `ParseHfConfig()`, `MapHuggingFaceWeightToTurboMind()`, `LoadWeightsFromSafetensors()`
  - `src/turbomind/models/linear_weight.cc` - `LinearWeight::prepare()` handles AWQ format conversion
  - `src/turbomind/core/data_format.cc` - `ResolveLinearWeightFormat()` for AWQ kUint4 format
- **Learnings:**
  - AWQ detection: `ParseHfConfig()` reads `quantization_config.quant_method == "awq"` from config.json
  - AWQ config: `CreateAwqLinearConfig()` sets `format = ResolveLinearWeightFormat(data_type, kUint4, awq_group_size, 1)`
  - Weight mapping: `.qweight` → `.weight`, `.qzeros` → `.zeros`, `.weight_scale` → `.scales`
  - Format conversion: `LinearWeight::prepare()` unpacks 4-bit weights, fuses scales/zeros
  - All linear layers (attention, FFN, MoE experts, DeltaNet) use `CreateAwqLinearConfig` when `is_awq=true`

---

