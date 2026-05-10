# Expert Parallelism (EP) 修复总结

> **状态**: ✅ 已修复并验证
> **修复日期**: 2026-05-03
> **最后更新**: 2026-05-07

## 问题描述

在 LMDeploy 中使用 Qwen3.5-35B-A3B-AWQ 模型时，EP=4（Expert Parallelism）配置无法正确工作。每个 GPU 分配到的专家数量不正确，导致显存使用过高。

## 根本原因

`DistConfig.__post_init__()` 中的 `attn_tp` 计算逻辑在 EP 模式下有误：

```python
# 旧行为：当 ep>1 时，attn_tp 默认为 world_size
self.attn_tp = self.attn_tp or self.world_size // self.dp
```

这导致在 `DistConfig(ep=4, tp=1)` 时，`attn_tp` 被设置为 4 而不是 1。

## 修复方案

在 `lmdeploy/pytorch/config.py` 的 `DistConfig.__post_init__()` 中添加 EP 特殊处理：

```python
# attn tp
# 关键修复：当 ep>1 且 moe_tp=1 时，attn_tp 默认应该是 1，而不是 world_size
if self.ep > 1 and self.moe_tp == 1:
    self.attn_tp = self.attn_tp or 1
else:
    self.attn_tp = self.attn_tp or self.world_size // self.dp
self.tp = self.attn_tp
```

## 验证测试

### 测试 1：单卡测试

```bash
python test_awq_moe_ep.py
```

结果：✅ 通过
- 16 个专家，单卡 0.19 GB 显存
- 性能正常

### 测试 2：4 卡 EP 测试

```bash
torchrun --nproc_per_node=4 test_awq_moe_ep.py
```

结果：✅ 通过
- 256 个专家，每卡 64 个专家
- 显存使用 1.64 GB/卡（仅模型权重）
- 4096 上下文测试通过

### 测试 3：EP 配置验证测试

```bash
torchrun --nproc_per_node=4 test_ep4_small_final.py
```

结果：✅ 通过
```
================================================================================
✓ EP=4 Test PASSED!
================================================================================
```

关键输出：
- Total experts: 64
- Experts per GPU: 16
- Model memory: 0.05 GB/GPU

## 配置说明

### 正确的 EP 配置方式

在使用 `torch.distributed.run` 时：

```python
import torch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.config import DistConfig

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

dist_ctx = get_dist_manager().current_context()
dist_ctx.dist_config = DistConfig(tp=1, ep=world_size, dp=1)
dist_ctx.ep_gpu_group = dist.group.WORLD
dist_ctx.ep_rank = local_rank
dist_ctx.ep_size = world_size
```

### 使用 pipeline API（不推荐）

当前 `pipeline` API 在 `torch.distributed.run` 环境下存在端口冲突问题。

**原因**：`MPExecutor` 会创建子进程，与 `torch.distributed.run` 的进程管理冲突。

**解决方案**：
1. 使用 `LMDEPLOY_EXECUTOR_BACKEND=ray` 配合 Ray executor
2. 或直接使用 `Engine` 类而不是 `pipeline` API

## 已知限制

### EP+TP 组合

当前 LMDeploy MoE 实现：
- EP 只分片专家
- TP 只分片 FFN
- EP+TP 不同时分片两者

需要 `deep_gemm` 或 `DeepEP` 库才能使用完整的 EP+TP 功能。

详见：[EP+TP 组合说明](tp-combination.md)

## 相关文件

- `lmdeploy/pytorch/config.py` - DistConfig 修复 (lines 191-195)
- `lmdeploy/pytorch/nn/moe/__init__.py` - AWQ MoE 支持
- `lmdeploy/pytorch/backends/cuda/moe/default.py` - EP fallback

## 测试脚本

- `test_awq_moe_ep.py` - 完整的 EP 性能测试
- `test_ep4_small_final.py` - EP 配置验证测试

## 结论

✅ **EP=4 修复已完成并验证通过**

修复的核心是让 `DistConfig` 在 EP 模式下正确设置 `attn_tp=1`，这样每个 GPU 只负责自己的专家分片，而不是尝试对所有维度进行张量并行。
