# C API InitFromPath HF 加载障碍分析

**Bead ID**: lmdeploy-pld
**日期**: 2026-05-19
**状态**: 分析完成

---

## 执行摘要

分析 C API `TM_TurboMind_InitFromPath()` 加载 HuggingFace 模型的障碍。实现已部分完成，但仍存在多个关键限制。

---

## 当前实现架构

### 初始化流程 (turbomind_c.cc:1226-1434)

```cpp
TM_TurboMind_InitFromPath()
├── CreateContext()          // 设置 CUDA context
├── CreateRoot()             // 创建 ModelRoot sentinel
├── Build Module Tree        // 构建完整模块树
│   ├── ModelWeight
│   │   ├── tok_embeddings (param)
│   │   ├── norm (NormWeight)
│   │   ├── output (LinearWeight)
│   │   └── layers (ModuleList[40])
│   │       └── [0..39] DecoderLayerWeight
│   │           ├── attention_norm (NormWeight)
│   │           ├── ffn_norm (NormWeight)
│   │           ├── attention (AttentionWeight)
│   │           └── feed_forward (FfnWeight)
├── LoadWeightsFromSafetensors()
├── ProcessWeights()         // GPU 内存分配 + prepare()
└── CreateEngine()           // 创建推理引擎
```

### 已实现功能

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| config.json 解析 | ✅ 完成 | 373-411 |
| AWQ quant_config 检测 | ✅ 完成 | 427-564 |
| safetensors 文件读取 | ✅ 完成 | 572-822 |
| 基础模块树构建 | ✅ 完成 | 1280-1399 |
| HF → TM 名称映射 | ✅ 完成 | 1074-1174 |
| 权重数据加载 | ✅ 部分 | 977-1072 |

---

## 关键障碍

### 1. AttentionWeight 子模块缺失 ⚠️

**问题**: 当前代码创建 `AttentionWeight` 作为单个模块，但没有创建其 LinearWeight 子模块。

**HF 结构**:
```
model.layers.0.self_attn.q_proj.weight   [num_heads * head_dim, hidden]
model.layers.0.self_attn.k_proj.weight   [num_kv_heads * head_dim, hidden]
model.layers.0.self_attn.v_proj.weight   [num_kv_heads * head_dim, hidden]
model.layers.0.self_attn.o_proj.weight   [hidden, num_heads * head_dim]
```

**TM 结构**:
```
layers.0.attention
├── w_qkv (LinearWeight)      // QKV fused
├── wo (LinearWeight)         // output projection
├── q_proj, q_a_proj, q_b_proj (LinearWeight)  // MoE/GQA variants
├── kv_a_proj (LinearWeight)
├── q_norm, k_norm, q_a_layernorm, kv_a_layernorm (NormWeight)
```

**障碍**: `LoadWeightsFromSafetensors()` 中的 `current->param(param_name)` 会失败，因为子模块不存在。

**需要修复**: 在创建 `AttentionConfig` 后，需要创建其子模块：
```cpp
// 需要创建但未实现的子模块
turbomind::core::LinearConfig q_proj_cfg, k_proj_cfg, v_proj_cfg;
q_proj_cfg.input_dim = hidden_size;
q_proj_cfg.output_dim = num_heads * head_dim;
// ... 还需要创建 w_qkv (fused)
```

### 2. QKV 融合未实现 ⚠️

**问题**: HF 模型有分离的 q_proj, k_proj, v_proj 张量，但 TM 使用 fused w_qkv。

**映射函数问题**:
```cpp
// 当前实现只做字符串替换
size_t q_proj_pos = result.find(".q_proj.");
if (q_proj_pos != std::string::npos) {
    result.replace(q_proj_pos, 9, ".q_proj.");  // 只是 q_proj -> q_proj，无变化
}
```

**需要**: 将三个 HF tensor 融合为一个 w_qkv tensor，形状：
```
w_qkv: [num_heads * head_dim + 2 * num_kv_heads * head_dim, hidden]
```

### 3. FfnWeight 子模块缺失 ⚠️

**问题**: FfnWeight 应该有 LinearWeight 子模块 (w1, w3, w2, w1w3)，但未创建。

**HF → TM 映射**:
```
model.layers.0.mlp.gate_proj.weight → layers.N.feed_forward.w1
model.layers.0.mlp.up_proj.weight   → layers.N.feed_forward.w3
model.layers.0.mlp.down_proj.weight → layers.N.feed_forward.w2
```

### 4. AWQ 量化处理不完整 ⚠️

**问题**: AWQ 模型有特殊的量化 tensor 结构：
```
model.weight           // INT4 packed in INT8, shape [hidden, intermediate/8]
model.weight_scale     // FP16, shape [intermediate/128, hidden]
model.weight_zero      // INT8, shape [intermediate/128, hidden]
```

**当前处理**:
```cpp
// LoadWeightsFromSafetensors() 只做简单的 memcpy
param.alloc(tensor_shape, tm_dtype);
std::memcpy(tensor.raw_data(), data, copy_size);  // 没有反量化！
```

**需要**: INT4 → FP16 反量化：
```cpp
// 需要添加 AWQ 反量化逻辑
for each INT4 weight:
    1. 读取 model.weight (INT4 packed)
    2. 读取 model.weight_scale (FP16)
    3. 读取 model.weight_zero (INT8)
    4. 计算: fp16_weight = (int4_weight - zero) * scale
```

### 5. DeltaNet/MoE 支持缺失 ⚠️

**问题**: Qwen3.6-35B-A3B 是 MoE 模型，包含 DeltaNet 层。

**缺失子模块**:
```
layers.N.linear_attn (DeltaNetWeight)
├── norm (NormWeight)
├── in_proj_all (LinearWeight)    // [3 * hidden, 2 * delta_dim]
├── out_proj (LinearWeight)
├── conv1d, A_log, dt_bias (params)
```

**需要配置检测**:
```cpp
// 检查 config.json 中是否有 MoE 配置
bool is_moe = config.get("num_local_experts", 0) > 1;
bool has_delta_net = config.get("use_linear_attn", false);
```

### 6. 嵌套 config 支持缺失

**问题**: Qwen3.5 MoE 的 config.json 结构：
```json
{
    "text_config": {  // 实际模型参数在这里
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        ...
    }
}
```

**当前解析**: 只读取顶层字段，无法获取嵌套的 text_config。

---

## 根本原因分析

### Python TurboMind vs C API 对比

| 组件 | Python (Builders) | C API |
|------|-------------------|-------|
| 模块创建 | `_tm.create_module(cfg)` 创建完整树 | `Module::create()` 只创建单层 |
| 子模块绑定 | Builders 自动创建并绑定 | 未实现 |
| 权重加载 | `_copy_shard_to_param()` 处理融合 | 只做简单的 memcpy |
| AWQ 处理 | QuantHandler 处理反量化 | 无 |
| MoE/DeltaNet | MoeBuilder, DeltaNetBuilder | 未实现 |

### C API 缺少的关键方法

1. **`_tm.create_module()` 调用** - 需要为每个子模块调用
2. **`add_child_raw()` 调用** - 绑定子模块到父模块
3. **权重融合逻辑** - QKV, FFN fusion
4. **AWQ 反量化** - INT4 → FP16

---

## 解决方案建议

### 方案 A: 增强 C API 实现

在 `TM_TurboMind_InitFromPath()` 中添加完整的子模块创建逻辑：

```cpp
// 增强后的流程
for each layer:
    // 创建 attention 子模块
    auto q_proj = Module::create(q_proj_cfg);
    auto k_proj = Module::create(k_proj_cfg);
    auto v_proj = Module::create(v_proj_cfg);
    auto w_qkv = FuseQKV(q_proj, k_proj, v_proj);  // 融合
    attention->add_child("w_qkv", w_qkv);
    attention->add_child("wo", wo);

    // 创建 ffn 子模块
    ffn->add_child("w1", w1);  // gate_proj
    ffn->add_child("w3", w3);  // up_proj
    ffn->add_child("w2", w2);  // down_proj

    // MoE 支持
    if (is_moe) {
        auto moe = Module::create(moe_cfg);
        layer->add_child("moe_ffn", moe);
    }

    // DeltaNet 支持
    if (has_delta_net) {
        auto delta = Module::create(delta_cfg);
        layer->add_child("linear_attn", delta);
    }
```

### 方案 B: 使用 Python Bridge 作为 fallback

当 C API 无法加载时，自动 fallback 到 Python bridge：
```cpp
if (InitFromPath() fails) {
    // 自动使用 Python bridge
    TM_InitFromPython(model_dir, session_len);
}
```

---

## 验收标准回顾

| 标准 | 当前状态 | 需要修复 |
|------|----------|----------|
| InitFromPath 不崩溃 | ⚠️ 部分 | 需添加子模块 |
| ProcessWeights 不崩溃 | ⚠️ 可能失败 | 需完整模块树 |
| 所有子模块存在 | ❌ 未实现 | 需要创建子模块 |
| AWQ 权重正确加载 | ❌ 未实现 | 需要反量化 |
| 推理请求成功 | ❌ 未知 | 需要端到端测试 |

---

## 下一步

1. **lmdeploy-6l2** (safetensors): 扩展 SafetensorsReader 处理 AWQ quantized tensors
2. **lmdeploy-82u** (module tree): 实现完整的子模块创建逻辑
3. **lmdeploy-sw3** (AWQ): 实现 INT4 → FP16 反量化
4. **lmdeploy-zk3** (e2e test): 端到端测试验证

---

## 参考文件

- `src/turbomind/capi/turbomind_c.cc` - C API 实现
- `lmdeploy/turbomind/builders/text_model.py` - Python Builder 模式
- `lmdeploy/turbomind/builders/attention.py` - Attention 子模块创建
- `src/turbomind/models/attention_weight.h` - AttentionWeight 结构
- `src/turbomind/models/decoder_layer_weight.h` - DecoderLayerWeight 结构
- `.ralph-tui/progress.md` - 之前的分析记录

---

*分析完成，等待任务分配进行修复。*