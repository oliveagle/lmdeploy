# ModelWeight 子模块树分析 - lmdeploy-5wx

## ModelWeight 完整结构树

```
ModelWeight
├── tok_embeddings_           (LinearWeight)        # 词嵌入层
├── norm_                      (NormWeight)          # 最终 LayerNorm
├── output_                    (LinearWeight)        # LM Head
├── norm_final_scale_          (Tensor)              # 可选: norm 的 scale
├── layers_                    (vector<DecoderLayerWeight>)  # 40 层 decoder
│   └── [0..num_layers-1]
│       ├── input_layernorm_    (NormWeight)         # 层输入归一化
│       ├── post_layernorm_     (NormWeight)         # 层后归一化 (可选)
│       ├── attention_          (AttentionWeight)    # 注意力
│       │   ├── qkv_proj_       (LinearWeight)       # QKV 投影
│       │   └── output_proj_    (LinearWeight)       # 注意力输出
│       ├── ffn_                (FfnWeight)           # 前馈网络
│       │   ├── w1_             (LinearWeight)       # FFN gate
│       │   ├── w2_             (LinearWeight)       # FFN output
│       │   └── w3_             (LinearWeight)       # FFN up
│       └── delta_net_          (DeltaNetWeight)     # DeltaNet (混合架构)
│           └── ... (SSM 相关权重)
└── shared_head_weight_        (LinearWeight)        # 共享头 (可选)
```

## DecoderLayerWeight 详细子模块

### AttentionWeight
- `qkv_proj_`: 包含 weight, bias, scale, zero 等 (AWQ 量化时有 scales/zeros)
- `output_proj_`: 注意力输出投影

### FfnWeight
- `w1_`: FFN gate 投影 (gate_proj)
- `w2_`: FFN output 投影 (down_proj)
- `w3_`: FFN up 投影 (up_proj)

### DeltaNetWeight
- 用于 Qwen3.6 等混合架构 (Attention + SSM)
- 包含 recurrent state 和 gating 权重

## C++ Module::create() 流程

在 `src/turbomind/models/model_weight.cc` 中:
1. `Module::create<ModelWeight>()` 创建根模块
2. 构造函数中 `add_child()` 注册所有子模块
3. `allocate_memory()` 分配内存
4. `set_param()` 设置配置参数

## C API 缺少的模块构建步骤

当前 C API `TM_TurboMind_InitFromPath()`:
- ✅ 创建 ModelWeight 根模块 (只有配置元数据)
- ❌ 不加载实际权重数据 (没有 tensor 内容)
- ❌ 不调用 SafetensorsReader 读取 .safetensors 文件
- ❌ 不调用 ModelLoader.export() 绑定权重到 C++ 运行时

## Python TurboMind 权重加载流程

1. `TurboMind.__init__()` → `_from_hf()` (lmdeploy/turbomind/turbomind.py:200)
2. 创建 `ModelLoader(model, model_comm, gpu_count, model_path, data_type, engine_config)`
3. `model_loader.export()` → `create_checkpoint()` → `model.model(Prefix(ckpt))`
4. `model.model()` 实际加载权重到 C++ 模块树

## 关键文件

| 文件 | 职责 |
|------|------|
| `src/turbomind/models/model_weight.h` | ModelWeight 类定义 |
| `src/turbomind/models/model_weight.cc` | ModelWeight 实现 |
| `src/turbomind/models/decoder_layer_weight.h` | DecoderLayerWeight 定义 |
| `src/turbomind/capi/turbomind_c.cc:580-714` | SafetensorsReader (未集成) |
| `lmdeploy/turbomind/model_loader.py` | Python 权重加载协调器 |
| `lmdeploy/turbomind/turbomind.py:~200` | _from_hf() 入口 |

## 验收标准完成

- ✅ 理解 ModelWeight 完整的子模块结构
- ✅ 找出 C API 缺少的模块构建步骤 (不加载权重数据，不调用 SafetensorsReader)
- ✅ 输出清晰的模块树结构文档
