# EP+TP 组合说明

## 核心问题

**4 张 16GB 卡无法运行 20GB 权重（AWQ 量化后）**

## 当前实现限制

LMDeploy MoE 实现中：
- **EP** 只分片专家（expert）
- **TP** 只分片 FFN 维度
- **EP+TP** 不同时分片两者

这意味着 EP+TP 组合无法有效减少显存使用。

## EP+TP 组合测试

| 配置 | 显存需求 | 结果 |
|------|----------|------|
| EP=4, TP=1 | ~18 GB/GPU | ❌ V100 16GB 不够 |
| EP=4, TP=4 | ~15 GB/GPU (MoE) + attention | ❌ 仍然不够 |
| EP=1, TP=4 | ~15 GB/GPU (MoE) + attention | ❌ 仍然不够 |

## 内存分析

### Qwen3.6-35B-A3B-AWQ 内存需求

| 组件 | 每 GPU 内存 | 说明 |
|------|----------|------|
| MoE 权重 (256/4=64 experts) | ~15 GB | EP=4 分片后每卡 64 专家 |
| KV cache (4K context) | ~1.5 GB | cache_max_entry_count=0.8 |
| 运行时开销 | ~1 GB | 中间结果、CUDA graph 等 |
| **总计** | **~18 GB** | **超过 V100 16GB** |

## 解决方案

1. **使用 A100 40GB 或更大显存的 GPU**
2. **使用 8 张 GPU（TP=8）进一步分片**
3. **测试更小的 MoE 模型**
4. **等待 `deep_gemm` 或 `DeepEP` 库集成**

## DistConfig 修复

之前 EP+TP 组合存在 `attn_tp` 计算错误，已在 `lmdeploy/pytorch/config.py` 中修复：

```python
if self.ep > 1 and self.moe_tp == 1:
    self.attn_tp = self.attn_tp or 1
else:
    self.attn_tp = self.attn_tp or self.world_size // self.dp
```

## 相关修复

- `9062dc93` - fix: DistConfig world_size and attn_tp for EP + TP combination
- `09445c9a` - fix: DistConfig attn_tp default for EP + TP combination
