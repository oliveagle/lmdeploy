# Qwen3.5 MoE AWQ 支持

> **状态**: ✅ 已修复
> **修复日期**: 2026-05-03
> **最后更新**: 2026-05-07

## 问题描述

Qwen3.5-35B-A3B-AWQ 模型在 LMDeploy 中加载时遇到多个问题：
1. AWQ 量化方法未在 MoE 模块中注册
2. `modules_to_not_convert` 层识别不正确
3. EP 模式下需要 Triton fallback

## 修复内容

### 1. 添加 AWQ 量化方法

**文件**: `lmdeploy/pytorch/nn/moe/__init__.py`

```python
from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

MOE_MODULE_MAP = {
    'awq': FusedMoEAWQ,
    # ... 其他量化方法
}
```

### 2. 修复层识别

**文件**: `lmdeploy/pytorch/config.py`

修复 `modules_to_not_convert` 中 MoE 层的识别，确保 `FusedMoE` 层不被转换为 AWQ。

### 3. 添加 EP fallback

**文件**: `lmdeploy/pytorch/backends/cuda/moe/default.py`

在 EP 模式下，当 fused_moe kernel 不可用时，自动 fallback 到 Triton 实现。

### 4. 权重加载修复

**文件**: `lmdeploy/pytorch/models/qwen3_5_moe.py`

修复 `__get_param` 函数，返回 `None` 而不是抛出异常，所有 `_load_weight_*` 函数添加 `if param is not None:` 检查。

## 内存需求

### Qwen3.6-35B-A3B-AWQ

| 配置 | 显存需求 | 说明 |
|------|----------|------|
| 单卡 | ~60 GB | 256 个专家 × 40 层 |
| EP=4 | ~18 GB/GPU | 64 个专家/卡 |
| EP=4 + TP=4 | ~15 GB/GPU (MoE) + attention | 仍然需要大显存 |

### 运行建议

- 使用 A100 40GB 或更大显存的 GPU
- 或使用 8 张 GPU（TP=8）
- 或测试更小的 MoE 模型

### 已知限制

当前 LMDeploy MoE 实现 EP+TP 不能有效减少内存：
- EP 只分片专家
- TP 只分片 FFN 维度
- 不同时分片两者

需要 `deep_gemm` 或 `DeepEP` 库才能使用完整的 EP 功能。

## 相关文件

- `lmdeploy/pytorch/nn/moe/__init__.py` - AWQ MoE 模块注册
- `lmdeploy/pytorch/config.py` - 层识别修复
- `lmdeploy/pytorch/backends/cuda/moe/default.py` - EP fallback
- `lmdeploy/pytorch/models/qwen3_5_moe.py` - 权重加载修复

## 相关文档

- [EP 修复总结](../ep/fix-summary.md)
- [EP+TP 组合说明](../ep/tp-combination.md)
