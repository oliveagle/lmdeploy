# ModelWeight 模块树结构文档

## 概述

`ModelWeight` 是 TurboMind 引擎的根权重模块，管理整个模型的权重树。Python 加载流程通过 Builder 模式逐步构建模块树并填充权重数据，C API 当前只创建了空壳 ModelWeight，缺少所有子模块和权重加载步骤。

## 完整的 ModelWeight 子模块树

```
ModelWeight (根)
├── tok_embeddings (Tensor param)          - 词嵌入查找表
├── output (LinearWeight)                  - LM Head (vocab_size x hidden)
│   ├── weight (Tensor)
│   └── bias (Tensor, 可选)
├── norm (NormWeight)                      - 最终 LayerNorm
│   └── weight (Tensor)
└── layers (ModuleList<DecoderLayerWeight>) - N 层 decoder layers
    └── [i] (DecoderLayerWeight)
        ├── attention_norm (NormWeight)
        │   └── weight (Tensor)
        ├── attention (AttentionWeight)
        │   ├── q_proj (LinearWeight)      - Q 投影
        │   │   ├── weight (Tensor)
        │   │   └── bias (Tensor, 可选)
        │   ├── k_proj (LinearWeight)      - K 投影
        │   │   ├── weight (Tensor)
        │   │   └── bias (Tensor, 可选)
        │   ├── v_proj (LinearWeight)      - V 投影
        │   │   ├── weight (Tensor)
        │   │   └── bias (Tensor, 可选)
        │   ├── o_proj (LinearWeight)      - 输出投影
        │   │   ├── weight (Tensor)
        │   │   └── bias (Tensor, 可选)
        │   ├── q_norm (NormWeight, 可选)   - QK Norm
        │   │   └── weight (Tensor)
        │   └── k_norm (NormWeight, 可选)   - QK Norm
        │       └── weight (Tensor)
        ├── ffn_norm (NormWeight)
        │   └── weight (Tensor)
        ├── feed_forward (FfnWeight)        - 标准 FFN (非 MoE)
        │   ├── w1 (LinearWeight)           - gate_proj (Silu 激活)
        │   ├── w2 (LinearWeight)           - down_proj
        │   └── w3 (LinearWeight)           - up_proj
        └── moe_ffn (MoeWeight, 可选)       - MoE FFN
            ├── gate (LinearWeight)
            └── experts (ModuleList<FfnWeight>)
                └── [e] (FfnWeight)
                    ├── w1 (LinearWeight)
                    ├── w2 (LinearWeight)
                    └── w3 (LinearWeight)
```

## 模块类型定义

### ModelWeight
- **文件**: `src/turbomind/models/model_weight.h`
- **子模块**: `output` (LinearWeight), `norm` (NormWeight), `layers` (ModuleList)
- **参数**: `tok_embeddings` (Tensor)
- **派生属性**: `data_type`, `hidden_units`, `vocab_size`, `num_layer` 等

### DecoderLayerWeight
- **文件**: `src/turbomind/models/decoder_layer_weight.h`
- **子模块**: `attention_norm`, `attention`, `ffn_norm`, `feed_forward` / `moe_ffn`

### AttentionWeight
- **文件**: `src/turbomind/models/attention_weight.h`
- **子模块**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`

### FfnWeight
- **子模块**: `w1`, `w2`, `w3` (均为 LinearWeight)

### LinearWeight
- **文件**: `src/turbomind/models/linear_weight.h`
- **参数**: `weight`, `bias`

### NormWeight
- **文件**: `src/turbomind/models/norm_weight.h`
- **参数**: `weight`

### ModuleList
- **文件**: `src/turbomind/core/module.h`
- 索引容器，存储 DecoderLayerWeight 数组

## Python 权重加载流程

### 1. 入口: `ModelLoader.export()` → `model.model(Prefix(ckpt))`

```
model_loader.py: ModelLoader.export()
  └→ self.model.model(Prefix(ckpt))
       └→ TextModelBuilder.model() [qwen3.py:61-75]
```

### 2. 构建模块树 (qwen3.py `model()` 方法)

```python
def model(self, pfx):
    root_cfg = make_model_weight_config(self.cfg)
    builder = TextModelBuilder(root_cfg, self._ctx, ...)
    builder.add_token_embeds(...)     # tok_embeddings Tensor
    builder.norm = self.norm(...)     # norm NormWeight
    builder.add_lm_head(...)          # output LinearWeight
    builder.layers = self.layers(...) # layers ModuleList
    builder.build()
```

### 3. Builder.build() 创建 C++ 模块

- `TextModelBuilder` → `create_module(ModelWeightConfig)` → ModelWeight
- `DecoderLayerBuilder` → `create_module(DecoderLayerConfig)` → DecoderLayerWeight
- `AttentionBuilder` → `create_module(AttentionConfig)` → AttentionWeight
- `FfnBuilder` → `create_module(FfnConfig)` → FfnWeight
- `ModuleListBuilder` → `create_module(ModuleListConfig)` → ModuleList

每个 Builder 调用 `ctx.add_module()` 或等效方法在 C++ 端创建对应的 Module 实例。

### 4. 权重数据加载

```python
def _linear(self, pfx):
    tensor = pfx.get('weight')   # 从 checkpoint 读取
    bias = pfx.get('bias')       # 可选
    return LinearTensor(tensor, bias)
```

- `pfx` 是 `Prefix` 对象，包装 checkpoint 文件读取
- `pfx.get('key')` 读取对应的权重 Tensor
- Builder 通过 `add_qkv_proj()`, `add_o_proj()`, `add_ffn()` 等方法将 Tensor 提交到 C++ Module

### 5. 后处理和验证

- `prepare()`: 递归调用所有子模块的后处理（格式转换、权重融合等）
- `verify()`: 遍历子树，收集未初始化的参数路径

## C API 缺少的步骤

对比 Python 流程，C API `InitFromPath()` 当前只做了一件事：

```cpp
// 当前状态: 只创建空壳 ModelWeight
ModelWeight::create(cfg);
```

**缺少的关键步骤**：

| 步骤 | Python 对应 | C API 现状 |
|------|-------------|------------|
| 1. 创建子模块树 | Builder.build() 逐个创建 | ❌ 完全缺失 |
| 2. 分配 Tensor 槽位 | pfx.get() + Builder 提交 | ❌ 完全缺失 |
| 3. 加载权重数据 | checkpoint 读取 | ❌ 完全缺失 |
| 4. 后处理 | prepare() | ❌ 未调用 |
| 5. 验证 | verify() | ❌ 未调用 |

## 下一步

要在 C API 中实现完整的权重加载，需要：

1. **模块树构建**: 实现 C++ 端的 `build_model_tree()` 函数，递归创建所有子模块
2. **权重加载**: 从模型文件（safetensors / .bin）读取权重数据并填充到对应 Tensor
3. **后处理**: 调用 `prepare()` 进行格式转换和融合
4. **验证**: 调用 `verify()` 确保所有权重正确加载

## 参考文件

- `src/turbomind/models/model_weight.h` - ModelWeight 定义
- `src/turbomind/models/model_weight.cc` - ModelWeight 实现
- `src/turbomind/models/decoder_layer_weight.h` - DecoderLayerWeight 定义
- `src/turbomind/models/linear_weight.h` - LinearWeight 定义
- `src/turbomind/models/norm_weight.h` - NormWeight 定义
- `src/turbomind/core/module.h` - Module 基类定义
- `lmdeploy/turbomind/model_loader.py` - Python 加载入口
- `lmdeploy/turbomind/models/qwen3.py` - Python 模型定义
- `lmdeploy/turbomind/builders/text_model.py` - TextModelBuilder
- `lmdeploy/turbomind/builders/decoder_layer.py` - DecoderLayerBuilder
- `lmdeploy/turbomind/builders/attention.py` - AttentionBuilder
- `lmdeploy/turbomind/builders/ffn.py` - FfnBuilder
- `lmdeploy/turbomind/builders/norm.py` - NormBuilder
- `lmdeploy/turbomind/builders/module_list.py` - ModuleListBuilder
