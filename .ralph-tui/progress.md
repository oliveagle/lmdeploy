# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- C++ 引擎通过 `GenerationConfig::output_last_hidden_state` 控制隐藏状态输出：`0` = 无，`1` = 全部token（kAll），`2` = 仅生成token（kGeneration，只返回最后一个token的隐藏状态）
- `output_last_hidden_state=kAll` 与 prefix_caching 不兼容，引擎会跳过这类请求并报错（见 `engine.cc:316`）
- Embedding 实现策略：对输入文本进行 forward pass，设置 `max_new_tokens=1` + `output_last_hidden_state=2`，从 `last_hidden_state` 输出张量提取最后一个token的向量，按 `dimensions` 参数截断
- `parse_hidden_size` 函数解析 `config.json` 时，需要处理嵌套配置（`text_config` 或 `model_config` 内）
- `spawn_blocking` 中使用的数据必须是 `'static` 生命周期，需要通过 `Arc::clone` 克隆 pool 引用

- `TM_TurboMind_InitFromHF` 纯 C++ 实现：当函数签名与 InitFromPath 功能相同时，直接委托调用 `TM_TurboMind_InitFromPath`，避免重复代码
- Rust FFI 签名必须严格匹配 C 头文件声明。`init_from_hf` 的签名从 `output_dir: &str` 改为 `trust_remote_code: bool, session_len: c_int`

---

## [2026-05-22] - lmdeploy-529
- 实现 TM_TurboMind_InitFromHF 纯 C++ 版本，移除 Python 桥接依赖
- **Files changed**:
  - `lmdeploy-rust-server/src/turbomind/capi/turbomind_c.cc` - 替换 Python 桥接为纯 C++ 调用 `TM_TurboMind_InitFromPath`
  - `lmdeploy-rust-server/src/turbomind_c.rs` - 修复 FFI 签名：`output_dir` → `trust_remote_code, session_len`，同步 Rust wrapper 方法签名
- **Learnings:**
  - `TM_TurboMind_InitFromHF` 与 `TM_TurboMind_InitFromPath` 功能相同（都解析 HF config.json + 加载 safetensors），所以 InitFromHF 直接委托给 InitFromPath 实现
  - Rust FFI 签名必须与 C 头文件 `include/turbomind_c.h` 完全一致，否则链接时会报签名不匹配

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

## [2026-05-22] - lmdeploy-tox
- 实现 C++ Engine Embeddings API，通过 `output_last_hidden_state` 获取文本嵌入向量
- **Files changed**:
  - `lmdeploy-rust-server/src/model/cpp_engine.rs`:
    - 新增 `hidden_size` 字段到 `TurboMindCEngine` 和 `ModelInfo`
    - 新增 `parse_hidden_size()` 函数从 config.json 解析 hidden_size
    - 实现 `TurboMindCEngine::embed()` 方法，使用 `output_last_hidden_state=2` 获取最后一个token的隐藏状态
    - 添加 `TM_SessionParam` 到 imports
  - `lmdeploy-rust-server/src/model/manager.rs`:
    - 更新 `ModelInfo` 初始化包含 `hidden_size` 字段
- **API 行为**:
  - 输入：`text: &str, dimensions: Option<usize>`
  - 输出：`Vec<f32>` - 文本的嵌入向量，默认长度为模型的 hidden_size，可按 dimensions 参数截断
  - 使用 `spawn_blocking` 在阻塞任务中运行 FFI 调用
  - 设置 `max_new_tokens=1`, `temperature=0.0`, `output_last_hidden_state=2`（kGeneration = 最后一个token）
  - 从 `last_hidden_state` 输出张量提取嵌入向量
- **Learnings:**
  - `output_last_hidden_state` 的值：`0`=无，`1`=全部tokens，`2`=仅最后一个token（kGeneration）
  - `spawn_blocking` 要求闭包捕获的数据是 `'static` 生命周期，需要 `Arc::clone` pool
  - config.json 中 hidden_size 可能在嵌套结构中（`text_config` 或 `model_config`）
  - C++ 引擎的 last_hidden_state 输出形状：单token为 `[hidden_dim]`，多token为 `[N, hidden_dim]`

---
