# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### Safetensors Weight Path Mapping (`MapHuggingFaceWeightToTurboMind`)

There are **two copies** of `turbomind_c.cc`:
1. `src/turbomind/capi/turbomind_c.cc` — main C++ library (canonical, most complete)
2. `lmdeploy-rust-server/src/turbomind/capi/turbomind_c.cc` — Rust server copy (may lag behind)

Always check both copies when modifying weight mapping logic.

**MTP (Multi-Token Prediction) mapping pattern**: `mtp.layers.X.*` → `layers.X.*` (alias, share weights with main model)
- Strip `mtp.` prefix, skip MTP-specific params (`mtp.norm`, `mtp.fc`, `mtp.pre_fc_norm_*`)
- Keep `layers.` path and apply standard mappings

---

## [2026-05-22] - lmdeploy-85l

### What was implemented
- 分析了 C++ 引擎 AWQ 加载失败的根因
- 定位了 `turbomind.cc:152` 的 data_type 检查逻辑
- 理解了 InitFromPath 权重加载流程
- 确认了 AWQ 权重格式与 C++ 引擎期望格式的差异

### Files analyzed
- `src/turbomind/turbomind.cc` - TurboMind::Impl 构造函数中的 data_type 检查
- `src/turbomind/engine/engine_config.h` - EngineConfig 结构定义
- `src/turbomind/core/data_type.h` - DataType 枚举定义
- `src/turbomind/capi/turbomind_c.cc` - C API 实现
- `lmdeploy-rust-server/src/model/cpp_engine.rs` - Rust C++ 引擎封装

### Learnings

#### 1. data_type 检查逻辑 (turbomind.cc:152)

```cpp
TurboMind::Impl::Impl(string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory):
    data_type_{}, engine_param_{}, ffi_ctx_factory_{ffi_ctx_factory}
{
    data_type_ = config.data_type;
    TM_CHECK(data_type_ == kBfloat16 || data_type_ == kHalf);
    // ...
}
```

**关键发现**：
- `data_type_` 是**激活/计算数据类型**，不是权重数据类型
- 检查只允许 `kBfloat16` (67591) 或 `kHalf` (66826)
- `kHalf` 是 `kFloat16` 的别名，值相同

#### 2. 数据类型编码系统

```cpp
// encode_data_type(sign, exponent, mantissa)
kFloat16  = (1 << 16) | (5 << 8) | 10 = 66826
kBfloat16 = (1 << 16) | (8 << 8) | 7  = 67591
kUint4    = (0 << 16) | (0 << 8) | 4  = 4
```

C API 到 C++ 的转换：
```cpp
// C API
TM_DATATYPE_FP16 = 10
TM_DATATYPE_BF16 = 13

// FromCDataType 转换
case TM_DATATYPE_FP16: return DT::kFloat16;  // 10 -> 66826
case TM_DATATYPE_BF16: return DT::kBfloat16; // 13 -> 67591
```

#### 3. AWQ 权重格式 vs 引擎期望

| 概念 | 数据类型 | 说明 |
|------|----------|------|
| **激活 dtype** | kHalf/kBfloat16 | GEMM 计算的数据类型 |
| **权重 dtype** | kUint4 (AWQ) | AWQ 量化权重存储格式 |
| **Scale dtype** | kHalf/kBfloat16 | AWQ 量化系数 |

**关键理解**：AWQ 模型的权重是 `kUint4` 格式，但：
1. 引擎的 `data_type_` (激活 dtype) 必须是 `kHalf` 或 `kBfloat16`
2. 权重的 `kUint4` 格式通过 `LinearWeight.format` 单独处理
3. Rust 代码正确设置了 `TM_DATATYPE_FP16` -> `kHalf`，检查应该通过

#### 4. InitFromPath 流程

```
TM_TurboMind_InitFromPath (turbomind_c.cc:1571)
├── CreateContext(index)        // 创建 CUDA 上下文
├── CreateRoot(index)           // 创建 ModelRoot sentinel
├── ParseHfConfig()             // 解析 HuggingFace config.json
├── Build ModelWeight tree      // 构建完整权重模块树
│   ├── tok_embeddings (Param)
│   ├── norm (NormWeight)
│   ├── layers (ModuleList)
│   │   └── decoder_layer (DecoderLayerWeight)
│   │       ├── attention_norm (NormWeight)
│   │       ├── attention (AttentionWeight)
│   │       │   └── w_qkv/w_o (LinearWeight, AWQ format)
│   │       └── ffn_norm + feed_forward (FfnWeight)
│   │           └── w1/w2/w3 (LinearWeight, AWQ format)
│   └── output (LinearWeight)
├── LoadWeightsFromSafetensors() // 加载权重数据
├── ProcessWeights(index)       // GPU 转移 + prepare()
└── CreateEngine(index)         // 创建推理引擎
```

#### 5. Rust C++ 引擎配置 (cpp_engine.rs)

```rust
// 正确：data_type 设置为 FP16（激活 dtype）
engine_config.set_data_type(TM_DataType::TM_DATATYPE_FP16);

// AWQ 检测和 quant_policy 设置
let is_awq = detect_awq_quantization(&model_path_obj);
let quant_policy = if is_awq { 4 } else { 0 };
engine_config.set_quant_policy(quant_policy);
```

**配置正确性**：
- ✅ `data_type` = FP16 (激活 dtype)
- ✅ `quant_policy` = 4 (AWQ)
- ✅ 权重的 AWQ 格式通过 `LinearConfig.format` 处理

### 结论

**data_type 检查失败的可能原因**：

1. **配置未正确传递** - `config.data_type` 在传递给构造函数前被修改或未初始化
2. **ABI 不匹配** - 编译的 `libturbomind_c.so` 与头文件定义不一致
3. **内存损坏** - `EngineConfig` 对象在传递过程中被破坏
4. **默认值问题** - `EngineConfig` 的默认 `data_type` 可能不是 `kHalf`

**推荐修复方向**：

1. **验证配置传递** - 在 `TM_EngineConfig_SetDataType` 中添加日志，确认 `data_type` 被正确设置
2. **检查编译一致性** - 确保 `libturbomind_c.so` 是从当前源码重新编译的
---

## [2026-05-22] - lmdeploy-gg4

### What was implemented
- Added `mtp.layers.*` alias support to Rust server's `turbomind_c.cc`
- Fixed QKV projection mapping (`.q_proj/.k_proj/.v_proj` → `.w_qkv` for fusion)
- Added MoE mappings (`.mlp.experts.` → `.moe_ffn.experts.`, `.mlp.gate.` → `.moe_ffn.gate.`)
- Added DeltaNet mappings (`.self_attn.in_proj.*` → `.linear_attn.in_proj_*`)
- Simplified code using single `size_t pos` variable

### Files changed
- `lmdeploy-rust-server/src/turbomind/capi/turbomind_c.cc` — Added MTP support and aligned with main `src/turbomind/capi/turbomind_c.cc`

### Learnings

#### 1. Dual `turbomind_c.cc` copies
There are two copies of the C API file:
- `src/turbomind/capi/turbomind_c.cc` — canonical version
- `lmdeploy-rust-server/src/turbomind/capi/turbomind_c.cc` — Rust server copy (may lag)

#### 2. MTP weight mapping pattern
- `mtp.layers.X.*` → `layers.X.*` (alias to main model layers)
- Skip MTP-specific params (`mtp.norm`, `mtp.fc`, `mtp.pre_fc_norm_*`) since C++ engine has no mtp module

#### 3. QKV fusion mapping
HF stores separate Q/K/V, TM uses fused `w_qkv`:
- `.q_proj.` → `.w_qkv.`
- `.k_proj.` → `.w_qkv.`
- `.v_proj.` → `.w_qkv.`

---
