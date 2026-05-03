# Qwen3.6-35B-A3B-AWQ MoE 模型分析报告

## 模型架构

| 属性 | 值 |
|------|-----|
| 隐藏维度 | 2048 |
| 专家数 | 256 |
| FFN 维度 | 512 |
| 层数 | 40 |
| 总参数量 | 8.75 B |

## 参数分布

| 组件 | 参数量 | 占比 | 内存 (float16) |
|------|--------|------|----------------|
| **MoE** | 5.81 B | 66.5% | 10.83 GB |
| **Attention** | 1.45 B | 16.6% | 2.71 GB |
| **其他** | 1.48 B | 16.9% | 2.76 GB |
| **总计** | 8.75 B | 100% | 16.29 GB |

## 内存需求分析

### 单张 GPU (无分片)
- **16.29 GB** (超过 16GB 限制)

### EP=2 (2 张 GPU)
- MoE: `10.83 / 2 = 5.42 GB`
- Attention: `2.71 GB`
- 其他: `2.76 GB`
- **总计: ~10.89 GB/卡** (超过 16GB 限制)

### EP=4 (4 张 GPU)
- MoE: `10.83 / 4 = 2.71 GB`
- Attention: `2.71 GB`
- 其他: `2.76 GB`
- **总计: ~8.18 GB/卡** (理论上可行)

### EP=2 + TP=2 (4 张 GPU)
- MoE: `10.83 / 4 / 2 = 1.36 GB` (EP 分片 + TP 分片)
- Attention: `2.71 / 2 = 1.36 GB` (TP 分片)
- 其他: `2.76 GB` (未分片)
- **总计: ~5.48 GB/卡** (最优方案)

## 代码修改

### 1. 支持 EP + TP 组合

**文件**: `lmdeploy/pytorch/nn/moe/default.py`

**修改内容**:
- 添加 `moe_tp_size` 参数到 `LinearWeights`
- 添加 `weight_loader_ep_tp` 方法处理 EP + TP 权重加载
- 修改 `FusedMoE.__init__` 支持 EP 和 TP 同时启用

### 2. 修改 DistConfig

**文件**: `lmdeploy/pytorch/config.py`

**修改内容**:
- 允许 `dp=1` 时也支持 `moe_tp`
- 修改 `world_size` 计算逻辑支持 EP + TP 组合

## 测试结果

### EP=4 配置
```python
PytorchEngineConfig(
    dtype='float16',
    cache_max_entry_count=0.01,
    eager_mode=True,
    tp=1,
    ep=4,
    empty_init=True,
    session_len=2048,
    max_prefill_token_num=1024,
)
```

**结果**: CUDA OOM (尝试分配 128 MiB 时失败)

**分析**: 
- 理论上 MoE 权重只需 2.71 GB/卡
- 但实际运行时 OOM，可能原因：
  - 中间激活值占用额外内存
  - KV Cache 占用内存
  - 模型构建时的临时张量
  - Ray 进程开销

## 结论

1. **LMDeploy 的 MoE 实现 EP 和 TP 原本互斥**，已修改为支持组合使用

2. **Qwen3.6-35B-A3B-AWQ 无法在单张 16GB GPU 上运行**，至少需要：
   - 4 张 16GB GPU (EP=4)，或
   - 8 张 16GB GPU (EP=4 + TP=2)

3. **建议**:
   - 使用更大的 GPU (24GB 或 48GB)
   - 增加更多 GPU 卡
   - 使用 CPU Offload
   - 考虑使用更小的模型或更激进的量化

## 相关文件

- `lmdeploy/pytorch/nn/moe/default.py` - MoE 实现
- `lmdeploy/pytorch/config.py` - 分布式配置
- `lmdeploy/pytorch/models/qwen3_5_moe.py` - Qwen3.6 MoE 模型
