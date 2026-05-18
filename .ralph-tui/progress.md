# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### SafetensorsReader Header-Only Pattern

**问题**: 需要在 C++ 层读取 HuggingFace safetensors 格式文件，原有实现在 turbomind_c.cc 中是简单的内联实现。

**解决方案**: 创建头文件-only safetensors 读取器 `src/turbomind/utils/safetensors_reader.h`

**关键特性**:
- Header-only 实现，无外部依赖
- 标准 safetensors 格式支持（8字节 header 大小 + JSON + 二进制数据）
- 支持多维 tensor shape 和各种 dtype（F32, F16, BF16, I32, I64, U8 等）
- 提供 `get_tensor_meta()` 获取元数据，`read_tensor()` 读取数据
- C API 包装通过 `TM_Safetensors_*` 函数系列暴露

**使用方式**:
```cpp
#include "src/turbomind/utils/safetensors_reader.h"

// 打开文件并解析 header
auto reader = SafetensorsReader("/path/to/model.safetensors");

// 获取 tensor 数量和名称
size_t n = reader.num_tensors();
const std::string& name = reader.tensor_name(0);

// 获取 tensor 元数据
const auto* meta = reader.get_tensor_meta("layer.weight");
if (meta) {
    std::cout << "dtype: " << (int)meta->dtype << "\n";
    std::cout << "shape: " << meta->shape[0] << "x" << meta->shape[1] << "\n";
}

// 读取 tensor 数据
std::vector<uint8_t> data = reader.read_tensor("layer.weight");
```

**C API 适配**: 通过 `ToCApiDtype()` 函数将内部 `TM_DataType` 转换为 C API 版本。

### HfConfigParser JSON Library Pattern

**问题**: HuggingFace config.json 解析需要支持嵌套结构（text_config, quantization_config），旧的简单行解析无法处理。

**解决方案**: 创建头文件-only JSON 解析器 `src/turbomind/utils/hf_config_parser.h`

**关键特性**:
- 无外部依赖（header-only）
- 支持嵌套对象路径查询（`get("text_config.hidden_size")`）
- 支持数组、布尔值、null、浮点数
- 完整的值类型（需要 copy/move 构造函数用于容器）

**使用方式**:
```cpp
#include "src/turbomind/utils/hf_config_parser.h"

auto root = turbomind::HfConfigParser::ParseFile("/path/to/config.json");
int hidden = root.get("hidden_size").as_int(4096);           // 顶层字段
int layers = root.get("text_config.num_hidden_layers").as_int(32);  // 嵌套字段
std::string method = root.get("quantization_config.quant_method").as_string("");
```

**注意事项**: `Value` 类需要完整的 copy/move 构造函数，因为 std::map/std::vector 需要可复制/可移动的元素。

### TurboMind C API Initialization Sequence

Pure C++ inference without Python uses this initialization sequence:

```rust
// 1. Create engine config
let mut engine_config = EngineConfig::new()?;
engine_config.set_session_len(65536);
engine_config.set_max_batch_size(32);
engine_config.set_cache_block_seq_len(64);
engine_config.set_enable_metrics(true);
engine_config.set_quant_policy(4);  // 4 for AWQ
engine_config.add_device(0);

// 2. Create TurboMind instance
let tm = TurboMind::create(model_path, &mut engine_config)?;

// 3. Initialize from model path (builds module tree, loads weights)
tm.init_from_path(device_id, model_path, trust_remote_code)?;

// 4. Create inference request
let request = ModelRequest::create(&tm)?;

// 5. Prepare input tensors
let mut input_tensors = TensorMap::new()?;
input_tensors.set_int64("input_ids", &ids, &shape);
input_tensors.set_int32("sequence_length", &len, &shape);

// 6. Prepare generation config
let mut gen_cfg = GenConfig::new()?;
gen_cfg.set_max_new_tokens(100);
gen_cfg.set_temperature(0.7);

// 7. Run inference
request.forward(&mut input_tensors, &session, &gen_cfg, false, true, &mut output_tensors)?;

// 8. Extract output
let (data_ptr, size) = request.get_output("output_ids")?;
let output_ids: Vec<i32> = unsafe {
    std::slice::from_raw_parts(data_ptr as *const i32, size / 4).to_vec()
};
```

### ModelWeight Module Tree Structure

The complete ModelWeight module tree is built by Python TurboMind Builders:

```
ModelWeight
├── tok_embeddings (param) - [vocab_size, hidden_size]
├── output (LinearWeight) - LM Head
│   └── weight, bias, scales, zeros (params)
├── norm (NormWeight) - Final RMS norm
│   └── weight (param)
└── layers (ModuleList)
    └── 0..N (DecoderLayerWeight)
        ├── attention_norm (NormWeight)
        │   └── weight (param)
        ├── ffn_norm (NormWeight)
        │   └── weight (param)
        ├── attention (AttentionWeight)
        │   ├── q_norm, k_norm, q_a_layernorm, kv_a_layernorm (NormWeight)
        │   └── w_qkv, wo, q_proj, q_a_proj, q_b_proj, kv_a_proj (LinearWeight)
        │       └── weight, bias, scales, zeros (params)
        ├── feed_forward (FfnWeight)
        │   └── w1, w3, w2, w1w3 (LinearWeight)
        │       └── weight, bias, scales, zeros (params)
        ├── linear_attn (DeltaNetWeight) - optional for Qwen3.5-MoE
        │   ├── norm (NormWeight)
        │   ├── in_proj_all, out_proj (LinearWeight)
        │   └── conv1d, A_log, dt_bias (params)
        └── moe_ffn (MoeWeight) - optional for MoE models
            ├── gate (LinearWeight)
            ├── experts (ModuleList of FfnWeight)
            ├── shared_experts (FfnWeight)
            └── routing (FfnWeight)
```

**Key Finding**: C API `InitFromPath()` only creates empty shell modules. The Python Builders (TextModelBuilder, AttentionBuilder, FfnBuilder, MoeBuilder, NormBuilder) are responsible for:
1. Creating child modules via `_tm.create_module(cfg)`
2. Committing weight data via `_copy_shard_to_param()`
3. Attaching children via `add_child_raw()`

---

## 2026-05-19 - lmdeploy-72b
- **Implemented**: Rust Server pure C++ inference engine (no Python dependency)
- **Files changed**:
  - `lmdeploy-rust-server/src/model/cpp_engine.rs` - New `TurboMindCEngine` struct that uses C API directly
  - `lmdeploy-rust-server/src/model/mod.rs` - Added `cpp_engine` module and exports
  - `lmdeploy-rust-server/src/turbomind_c.rs` - Added `set_cache_block_seq_len()`, `get_output()`, `process_weights()`, `create_engine()` wrapper methods
  - `lmdeploy-rust-server/src/config.rs` - Added `engine_type` field to `ModelConfig` (default: "python_bridge")
  - `lmdeploy-rust-server/config/default.toml` - Added `engine_type = "python_bridge"` default
- **Learnings**:
  - C API `TM_TurboMind_InitFromPath()` builds complete ModelWeight module tree from safetensors files
  - Module tree creation: CreateContext -> CreateRoot -> Build ModelWeight children -> ProcessWeights -> CreateEngine
  - Output tensors extracted via `TM_ModelRequest_GetOutput()` after forward inference
  - Engine type selection: `python_bridge` (compatible) vs `pure_cpp` (no Python dep)
  - C API weight loading uses `SafetensorsReader` with simple JSON parsing (no external JSON lib)
  - AWQ quantization auto-detected from config.json (`quantization_config.quant_method == "awq"`)
  - `TM_EngineConfig_SetCacheBlockSeqLen()` exists in C API but was missing from Rust FFI wrapper
  - The `set_quant_policy()` config controls quantization (0=none, 4=AWQ 4-bit, 8=KV cache INT8)

---

## 2026-05-18 - lmdeploy-5wx
- **Implemented**: Analysis of Python TurboMind weight loading and ModelWeight module tree
- **Files changed**: `.ralph-tui/progress.md` - Documented complete ModelWeight module tree structure
- **Learnings**:
  - Python TurboMind uses Builder pattern (TextModelBuilder, AttentionBuilder, FfnBuilder, MoeBuilder, NormBuilder)
  - Builders create child modules via `_tm.create_module(cfg)` and commit weight data via `_copy_shard_to_param()`
  - Children are attached via `add_child_raw()` after weight data is committed
  - C API `InitFromPath()` only creates empty shell modules without actual weight data
  - The missing piece in C API is that it doesn't build the LinearWeight children (w_qkv, wo, w1, w2, w3) with their params
  - AttentionWeight has 9 LinearWeight children (w_qkv, wo, q_proj, q_a_proj, q_b_proj, kv_a_proj) + 5 NormWeight children (q_norm, k_norm, q_a_layernorm, kv_a_layernorm)
  - FfnWeight has 4 LinearWeight children (w1, w3, w2, w1w3)
  - DeltaNetWeight has 2 LinearWeight children (in_proj_all, out_proj) + 1 NormWeight + 3 params (conv1d, A_log, dt_bias)

---

## 2026-05-19 - lmdeploy-qp8
- **Implemented**: Rust Server SSE Streaming 支持
- **Files changed**:
  - `lmdeploy/turbomind/python_bridge.py` - Added `generate_stream()` async generator and command handler
  - `lmdeploy-rust-server/src/model/python_bridge.rs` - Added `GenerateStream` command, `BridgeStreamChunk` type, `generate_stream()` method
  - `lmdeploy-rust-server/src/model/engine.rs` - Implemented `generate_stream()` to use Python bridge streaming
- **Learnings**:
  - Python TurboMind's `async_stream_infer()` with `stream_output=True` yields tokens one-by-one
  - Need to use `async for` with `yield` in Python async generators (not `return` with values)
  - Rust streaming uses `mpsc::channel` + `ReceiverStream` to bridge sync thread reading to async Stream
  - The HTTP handler was already wired up; just needed to implement the underlying `generate_stream()` method
  - SSE streaming sends `data: {json}` lines with `event: chat.completion.chunk` prefix
  - TTFT (Time To First Token) is tracked by recording time when first token is received
  - Use `filter_map` to skip empty tokens in the stream before sending to clients

---

## 2026-05-18 - lmdeploy-jpv
- **Implemented**: Fixed weight_serializer.py for nested config handling (Qwen3.5 MoE)
- **Files changed**:
  - `lmdeploy/turbomind/weight_serializer.py` - Added nested config support (text_config, model_config)
  - `tests/test_weight_serializer.py` - Created tests for serialization
- **Learnings**:
  - Qwen3.5 MoE models have nested config structure - actual model params are in `text_config` sub-object
  - Weight serializer must handle both flat configs (standard HF) and nested configs (multimodal/MoE models)
  - Safetensors I/O uses 8-byte header size + JSON header + binary data format
  - Tensor alignment to 64-byte boundaries is important for GPU loading performance
  - The Python weight_serializer.py is fully functional for HF→TM .bin conversion
  - C++ weight_serializer.cc is still a placeholder - uses Python bridge for actual conversion

---

## 2026-05-19 - lmdeploy-pld
- **Implemented**: 分析 C API InitFromPath 限制，找出 HF 加载障碍
- **Files changed**:
  - `ANALYSIS_INITFROMPATH_LIMITS_20260519.md` - 创建详细分析报告
- **Learnings**:
  - C API InitFromPath 已实现基础模块树构建，但缺少关键子模块
  - AttentionWeight 需要 LinearWeight 子模块 (w_qkv, wo, q_proj, k_proj, v_proj)
  - FfnWeight 需要 LinearWeight 子模块 (w1, w3, w2, w1w3)
  - QKV 融合未实现：HF 有分离的 q/k/v_proj，TM 使用 fused w_qkv
  - AWQ 量化处理不完整：INT4 → FP16 反量化缺失
  - DeltaNet/MoE 支持缺失：Qwen3.6-35B-A3B 需要 linear_attn 子模块
  - 嵌套 config 支持缺失：Qwen3.5 MoE 的 text_config 未解析
  - Python Builder 模式 vs C API：Python 使用 `_tm.create_module()` + `add_child_raw()`，C API 只实现单层
  - 当前 LoadWeightsFromSafetensors() 只做 memcpy，无权重格式转换
  - 关键方法缺失：权重融合逻辑、AWQ 反量化、子模块绑定

---


## 2026-05-19 - lmdeploy-uq0
- **Implemented**: 在 C++ 层实现 HuggingFace config.json 解析器
- **Files changed**:
  - `src/turbomind/utils/hf_config_parser.h` - 新增 header-only JSON 解析器
  - `src/turbomind/utils/test_hf_config_parser.cc` - 单元测试
  - `src/turbomind/capi/turbomind_c.cc` - 使用新解析器替换旧的行解析
- **Learnings**:
  - `Value` 类需要完整的 copy/move 构造函数，因为 std::vector 需要可复制元素
  - 嵌套 config 支持（text_config, model_config）通过 lambda + 引用返回实现
  - HfModelConfig 结构统一管理所有模型配置字段
  - ParseHfConfig() 使用 helper lambda (get_int, get_string, get_bool) 简化代码
  - MoE 和 DeltaNet 配置从 config.json 自动检测（num_local_experts, use_linear_attn）
  - AWQ quantization 从 quantization_config.quant_method == "awq" 检测


## 2026-05-19 - lmdeploy-6l2
- **Implemented**: 在 C++ 层实现 safetensors 文件读取
- **Files changed**:
  - `src/turbomind/utils/safetensors_reader.h` - 新增 header-only safetensors 读取器
  - `src/turbomind/capi/turbomind_c.cc` - 更新为使用新的 safetensors_reader.h
  - `src/turbomind/utils/test_safetensors_reader.cc` - 单元测试
- **Learnings**:
  - Safetensors 格式：8字节 header 大小（小端序）+ JSON metadata + 二进制 tensor 数据
  - JSON parsing 需要处理嵌套结构（tensor → dtype/shape/data_offsets）
  - 使用 std::vector<uint8_t> 作为通用数据容器，避免内存泄漏
  - C API 适配需要内部 TM_DataType 到外部 TM_DataType 的转换
  - LoadWeightsFromSafetensors 直接使用 C++ API 避免 C API 分配/释放循环
  - 移动语义（std::move）优化 vector 返回值
  - Thread-local 存储 用于返回临时字符串指针
  - __metadata__ 条目需要特殊处理（跳过）

---

