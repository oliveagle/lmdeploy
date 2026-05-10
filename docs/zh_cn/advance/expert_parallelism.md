# 专家并行 (EP)

专家并行 (Expert Parallelism, EP) 是一种针对混合专家 (MoE) 模型的并行策略，它将不同的专家分布到不同的 GPU 上。这与张量并行 (TP) 不同，TP 是将单个张量分片到多个 GPU 上。

## 概述

在 MoE 模型中，每个层包含多个专家网络。EP 允许你：

- **降低每 GPU 内存**：每个 GPU 只存储部分专家
- **扩展更大模型**：运行单 GPU 无法容纳的更多专家模型
- **保持吞吐量**：专家路由允许高效的 GPU 利用率

### EP vs TP

| 特性 | 专家并行 (EP) | 张量并行 (TP) |
|------|--------------|--------------|
| **分片粒度** | 按专家分片 | 按张量分片 |
| **通信模式** | All-to-All（专家路由） | All-Reduce（每层） |
| **适用场景** | 多专家 MoE 模型 | 稠密模型或少专家 MoE |
| **内存效率** | 高（每 GPU 存储 1/EP 专家） | 中（每 GPU 存储所有专家） |
| **延迟影响** | 低（仅在专家路由时） | 较高（每层 all-reduce） |

## 配置

### 基本用法

```python
from lmdeploy import TurbomindEngineConfig, pipeline

# 使用 EP=4 创建引擎配置
engine_config = TurbomindEngineConfig(
    ep=4,              # 专家并行大小
    tp=1,              # 张量并行大小
    device_num=4,      # GPU 总数 (ep * tp)
    session_len=2048,  # 会话长度
    max_batch_size=1,  # 最大批处理量
    quant_policy=8,    # KV 缓存量化 (4+8 bit 适用于 MoE 模型)
)

# 创建 pipeline
pipe = pipeline(
    model_path='/path/to/qwen3.6-35b-a3b-awq',
    backend='turbomind',
    engine_config=engine_config,
)

# 运行推理
response = pipe(['你好，你怎么样？'])
print(response[0])
```

### 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `ep` | int | 1 | 专家并行的 GPU 数量。必须能被 `device_num` 整除。 |
| `ep_rank` | int | 0 (自动) | 当前 GPU 在 EP 组中的 rank (0 到 ep-1)。根据设备 ID 自动计算。 |
| `tp` | int | 1 | 张量并行的 GPU 数量。必须能被 `device_num` 整除。 |
| `device_num` | int | 1 | GPU 总数。必须等于 `ep * tp`。 |

### 专家分布

对于 256 个专家的 MoE 模型，EP=4 时：

| GPU (Rank) | 专家范围 | 专家数量 |
|------------|----------|----------|
| 0 | [0, 63] | 64 |
| 1 | [64, 127] | 64 |
| 2 | [128, 191] | 64 |
| 3 | [192, 255] | 64 |

专家范围计算公式：
```python
experts_per_rank = (total_experts + ep_size - 1) // ep_size
ep_first_expert = ep_rank * experts_per_rank
ep_num_experts = min(experts_per_rank, total_experts - ep_first_expert)
```

## 性能特征

### 内存使用

Qwen3.6-35B-A3B-AWQ (256 专家) 的每 GPU 内存使用：

| 配置 | 每 GPU 内存 | 总内存 |
|------|------------|--------|
| EP=4, TP=1 | ~15 GB | ~60 GB |
| TP=4, EP=1 | ~18 GB | ~72 GB |

**关键洞察**: EP 降低了每 GPU 的内存，因为每个 GPU 只存储 1/EP 的专家。

### 吞吐量和延迟

| 指标 | EP=4, TP=1 | TP=4, EP=1 |
|------|-----------|-----------|
| **吞吐量** | 相似或更好 | 基线 |
| **延迟** | 略高 (all_reduce) | 较低 (无 all_reduce) |
| **批处理能力** | 更高 (内存更少) | 较低 (内存更多) |

**权衡**:
- EP 由于专家路由中的 all_reduce 通信，延迟略高
- EP 由于内存减少，允许更大的批处理量
- 对于 MoE 模型，EP 通常提供比 TP 更好的整体吞吐量

## 支持的模型

EP 支持 Turbomind 后端的 MoE 模型，包括：

- Qwen3.5-27B-AWQ
- Qwen3.6-35B-A3B-AWQ
- 其他 Qwen MoE 变体

## 硬件需求

### 最低配置

- **4x V100 16GB** (EP=4, TP=1)
  - 模型: Qwen3.6-35B-A3B-AWQ
  - 会话长度: 2048
  - 批处理量: 1
  - KV 量化: 4+8 bit

### 推荐配置

- **4x A100 40GB** (EP=4, TP=1)
  - 支持更长会话 (4096+)
  - 更大批处理量
  - 生产就绪性能

## 故障排查

### 显存不足 (OOM)

如果遇到 OOM 错误：

1. **减少会话长度**: `session_len=2048` → `session_len=1024`
2. **启用 KV 量化**: `quant_policy=8` (4+8 bit)
3. **减少批处理量**: `max_batch_size=1` (已最低)
4. **使用更大显存的 GPU**: A100 40GB 替代 V100 16GB

### 乱码输出

如果看到乱码文本（如全是 '!' 字符）：

1. **验证后端**: 使用 `backend='turbomind'`（而非 'pytorch'）
2. **检查 EP 配置**: 确保 `ep` 和 `device_num` 正确
3. **验证模型**: 确保使用支持的 MoE 模型

**已知问题**: PyTorch 后端 + EP=4 会产生乱码输出。请使用 Turbomind 后端。

### 配置错误

常见配置错误：

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `device_num != ep * tp` | 无效的并行配置 | 确保 `device_num = ep * tp` |
| `ep > device_num` | EP 大小超过可用 GPU | 减少 `ep` 或增加 `device_num` |
| `device_num` 与 GPU 不匹配 | 配置和硬件不匹配 | 将 `device_num` 设置为实际 GPU 数量 |

## 高级主题

### EP + TP 组合

当前 LMDeploy 支持：
- **EP=4, TP=1**: 纯专家并行（推荐用于 MoE 模型）
- **EP=1, TP=4**: 纯张量并行（基线）

**注意**: EP + TP 组合（如 EP=2, TP=2）尚未在 Turbomind 后端完成验证。

### 多节点 EP

多节点 EP 配置尚未测试。单节点 EP 已完全支持。

### KV 缓存量化

对于 MoE 模型，使用 `quant_policy=8`（4-bit K, 8-bit V）降低内存：

```python
engine_config = TurbomindEngineConfig(
    ep=4,
    tp=1,
    quant_policy=8,  # 4+8 bit KV 量化
)
```

## 参数传递链路

EP 参数从 Python 到 C++ 的完整传递链路：

```
TurbomindEngineConfig.ep (Python)
  → converter.py (设置 mlp_ep_size, mlp_ep_rank)
    → TurbomindModelConfig.model_config (写入 config.yaml)
      → turbomind.cc (读取 mlp_ep_size, mlp_ep_rank)
        → EngineParam (传递给 MoeParam)
          → LlamaDenseWeight (仅创建本地专家)
            → MoeFfnLayer (推理时执行 EP all_reduce)
```

## 参考资料

- [TurboMind 配置](./turbomind_config.md)
- [KV 量化](../quantization/kv_quant.md)
- [支持的 MoE 模型](../supported_models/supported_models.md)
