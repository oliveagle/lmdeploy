# TurboMind ModelWeight Module Tree Structure

## 概述

本文档描述了 TurboMind C++ 引擎中 ModelWeight 的完整子模块树结构，以及 Python 权重加载流程如何构建和填充此树。

## 核心机制

### Module 系统

- `Module` 是类型擦除的层次化模块基类
- 子类使用 X-macro (`TM_MODULE_DECLARE`) 声明 children 和 params
- Python 通过 `_tm.create_module(config)` 创建模块
- `add_child_raw()` 附加子模块到父模块

### Config 驱动创建

```cpp
// 每个 Module 类型都有对应的 Config
struct ModelWeightConfig: ModuleConfig {
    int tp_size;
    int tp_rank;
    DataType data_type;
    int hidden_units;
};
```

---

## 完整模块树结构

```
ModelRoot (sentinel, 每个 GPU 一个)
└── text_model: ModelWeight
    ├── tok_embeddings: Tensor (param)
    ├── output: LinearWeight (child)
    │   ├── weight: Tensor (param)
    │   ├── bias: Tensor (param, optional)
    │   ├── scales: Tensor (param, quantized)
    │   └── zeros: Tensor (param, quantized)
    ├── norm: NormWeight (child)
    │   └── weight: Tensor (param)
    └── layers: ModuleList (child)
        └── [0..N-1]: DecoderLayerWeight (indexed children)
            ├── attention_norm: NormWeight
            │   └── weight: Tensor
            ├── ffn_norm: NormWeight
            │   └── weight: Tensor
            ├── attention: AttentionWeight (optional)
            │   ├── w_qkv: LinearWeight
            │   │   ├── weight: Tensor
            │   │   ├── bias: Tensor (optional)
            │   │   ├── scales: Tensor (quantized)
            │   │   └── zeros: Tensor (quantized)
            │   ├── wo: LinearWeight
            │   │   └── ...
            │   ├── q_proj: LinearWeight (MLA)
            │   ├── q_a_proj: LinearWeight (MLA)
            │   ├── q_b_proj: LinearWeight (MLA)
            │   ├── kv_a_proj: LinearWeight (MLA)
            │   ├── q_norm: NormWeight (Qwen3)
            │   ├── k_norm: NormWeight (Qwen3)
            │   ├── q_a_layernorm: NormWeight (Qwen3.5)
            │   ├── kv_a_layernorm: NormWeight (Qwen3.5)
            │   └── sinks: Tensor (param, optional)
            ├── linear_attn: DeltaNetWeight (可选, Qwen3)
            │   ├── in_proj_qkv: LinearWeight
            │   ├── in_proj_z: LinearWeight
            │   ├── in_proj_a: LinearWeight
            │   ├── in_proj_b: LinearWeight
            │   ├── in_proj_all: LinearWeight
            │   ├── out_proj: LinearWeight
            │   ├── norm: NormWeight
            │   ├── conv1d: Tensor (param)
            │   ├── A_log: Tensor (param)
            │   └── dt_bias: Tensor (param)
            ├── feed_forward: FfnWeight
            │   ├── w1: LinearWeight
            │   ├── w3: LinearWeight
            │   ├── w2: LinearWeight
            │   └── w1w3: LinearWeight (fused, 可选)
            └── moe_ffn: MoeWeight (可选, MoE 模型)
                ├── shared_gate: LinearWeight
                ├── experts: ModuleList
                │   └── [0..E-1]: FfnWeight
                └── score_correction_bias: Tensor (param)
```

---

## X-macro 定义 (C++)

### ModelWeight

```cpp
// model_weight.h
#define MODEL_WEIGHT_CHILDREN(X)    \
    X(LinearWeight, output)         \
    X(NormWeight, norm)             \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X) X(tok_embeddings)
```

### DecoderLayerWeight

```cpp
// decoder_layer_weight.h
#define DECODER_LAYER_WEIGHT_CHILDREN(X)    \
    X(AttentionWeight, attention)           \
    X(DeltaNetWeight, linear_attn)          \
    X(FfnWeight, feed_forward)              \
    X(MoeWeight, moe_ffn)                   \
    X(NormWeight, attention_norm)           \
    X(NormWeight, ffn_norm)

#define DECODER_LAYER_WEIGHT_PARAMS(X)  // 空
```

### AttentionWeight

```cpp
// attention_weight.h
#define ATTENTION_WEIGHT_CHILDREN(X)    \
    X(LinearWeight, w_qkv)              \
    X(LinearWeight, wo)                 \
    X(LinearWeight, q_proj)             \
    X(LinearWeight, q_a_proj)           \
    X(LinearWeight, q_b_proj)           \
    X(LinearWeight, kv_a_proj)          \
    X(NormWeight, q_norm)               \
    X(NormWeight, k_norm)               \
    X(NormWeight, q_a_layernorm)        \
    X(NormWeight, kv_a_layernorm)

#define ATTENTION_WEIGHT_PARAMS(X) X(sinks)
```

### DeltaNetWeight

```cpp
// delta_net_weight.h
#define DELTA_NET_WEIGHT_CHILDREN(X)    \
    X(LinearWeight, in_proj_qkv)        \
    X(LinearWeight, in_proj_z)          \
    X(LinearWeight, in_proj_a)          \
    X(LinearWeight, in_proj_b)          \
    X(LinearWeight, in_proj_all)        \
    X(LinearWeight, out_proj)           \
    X(NormWeight, norm)

#define DELTA_NET_WEIGHT_PARAMS(X)      \
    X(conv1d)                           \
    X(A_log)                            \
    X(dt_bias)
```

### FfnWeight

```cpp
// ffn_weight.h
#define FFN_WEIGHT_CHILDREN(X)  \
    X(LinearWeight, w1)         \
    X(LinearWeight, w3)         \
    X(LinearWeight, w2)         \
    X(LinearWeight, w1w3)

#define FFN_WEIGHT_PARAMS(X)  // 空
```

---

## Python 权重加载流程

### 入口: ModelLoader.export()

```python
# lmdeploy/turbomind/model_loader.py
class ModelLoader:
    def export(self):
        ckpt = create_checkpoint(self.model_path, ...)
        self.model.model(Prefix(ckpt))  # 调用模型类的 model() 方法
        ckpt.close()
```

### LlamaModel.model() 典型流程

```python
# lmdeploy/turbomind/models/llama.py
def model(self, pfx):
    # 1. 创建 ModelWeight builder
    builder = TextModelBuilder(root_cfg, self._ctx, ...)

    # 2. 提交 token embeddings (Tensor param)
    builder.add_token_embeds(pfx.get('model.embed_tokens.weight'))

    # 3. 提交 final norm (NormWeight child)
    builder.norm = self.norm(pfx + 'model.norm')

    # 4. 提交 LM head (LinearWeight child)
    builder.add_lm_head(self._linear(pfx + 'lm_head'))

    # 5. 创建并提交所有 decoder layers
    builder.layers = self.layers(pfx + 'model.layers')

    # 6. build() 触发 C++ 模块创建和权重提交
    builder.build()
```

### layers() 构建流程

```python
def layers(self, pfx):
    layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
    for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
        d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
        d.attention_norm = self.norm(p + 'input_layernorm')
        d.attention = self.attn(p + 'self_attn')
        d.ffn_norm = self.norm(p + 'post_attention_layernorm')
        d.feed_forward = self.ffn(p + 'mlp')
        layers[i] = d.build()
    return layers.build()
```

---

## Builder.build() 关键步骤

```python
# lmdeploy/turbomind/builders/_base.py
def build(self) -> BuiltModule:
    if self._built:
        return BuiltModule(self._handles)

    # 1. 创建每个 GPU 的 C++ 模块
    self._create_handles()  # 调用 _tm.create_module(config)

    self._built = True

    # 2. 附加所有子模块
    for name, handles in self._pending_children.items():
        self._commit_child(name, handles)  # parent.add_child_raw(name, child)

    # 3. 提交所有 tensor 参数
    for name, (tensor, split_side) in self._pending_tensors.items():
        self._commit_tensor(name, tensor, split_side)  # param.alloc + copy_from

    return BuiltModule(self._handles)
```

---

## C API 缺少的步骤

### 当前状态: InitFromPath()

```cpp
// src/turbomind/capi/turbomind_c.cc
// InitFromPath() 只创建了空的 ModelWeight，没有:
// 1. 创建子模块树 (layers, attention, ffn 等)
// 2. 分配 tensor 内存
// 3. 复制权重数据
```

### 缺少的构建步骤

| 步骤 | Python 流程 | C API 缺失 |
|------|------------|-----------|
| 1. 子模块创建 | `_tm.create_module(config)` 循环调用 | 无 |
| 2. 子模块附加 | `add_child_raw(name, child)` | 无 |
| 3. Tensor 分配 | `param.alloc(shape, dtype)` | 无 |
| 4. 权重复制 | `dst.copy_from(tensor)` | 无 |

### 需要实现的功能

1. **模块树创建**:
   - 创建 40 个 DecoderLayerWeight
   - 每个 layer 创建 attention/ffn/norm 子模块
   - 创建 output LinearWeight 和 norm NormWeight

2. **权重加载**:
   - 从 safetensors 读取权重
   - 根据 TP 配置分片权重
   - 分配 GPU 内存并复制

3. **配置填充**:
   - head_dim, kv_head_num 等从第一层提取
   - vocab_size, embedding_size 从 tok_embeddings 提取

---

## 参考文件

| 文件 | 描述 |
|------|------|
| `src/turbomind/models/model_weight.h` | ModelWeight 定义 |
| `src/turbomind/models/decoder_layer_weight.h` | DecoderLayerWeight 定义 |
| `src/turbomind/models/attention_weight.h` | AttentionWeight 定义 |
| `src/turbomind/models/ffn_weight.h` | FfnWeight 定义 |
| `src/turbomind/models/delta_net_weight.h` | DeltaNetWeight 定义 |
| `src/turbomind/core/module.h` | Module 基类和 X-macro |
| `lmdeploy/turbomind/builders/_base.py` | Builder 基类 |
| `lmdeploy/turbomind/builders/text_model.py` | TextModelBuilder |
| `lmdeploy/turbomind/models/llama.py` | Llama 权重加载示例 |
