# V100 GPU 限制说明

## 核心问题

**V100 16GB 无法运行 Qwen3.6-35B-A3B-AWQ 模型**

## 内存分析

| 组件 | 每 GPU 内存 | 说明 |
|------|----------|------|
| MoE 权重 (256/4=64 experts) | ~15 GB | EP=4 分片后每卡 64 专家 |
| KV cache (4K context) | ~1.5 GB | cache_max_entry_count=0.8 |
| 运行时开销 | ~1 GB | 中间结果、CUDA graph 等 |
| **总计** | **~18 GB** | **超过 V100 16GB** |

## GPU 兼容性

| GPU | 显存 | 状态 |
|-----|------|------|
| V100 16GB | 16 GB | ❌ 不够 |
| V100 32GB | 32 GB | ✅ 可用 |
| A100 40GB | 40 GB | ✅ 推荐 |
| A100 80GB | 80 GB | ✅ 最佳 |

## TileLang 兼容性

V100 + CUDA 12.8 存在 TileLang 兼容性问题：

```
RuntimeError: No suitable user-defined conversion from "__nv_bfloat16" to "__half" exists
```

**解决**: 使用 A100/H100 或降级 CUDA。

## CPU Offload 限制

❌ **CPU offload 无法解决模型权重内存问题**

- LMDeploy 的 CPU offload 只支持 KV cache
- 模型权重必须常驻 GPU 内存
- Qwen3.5 MoE AWQ 权重太大 (15 GB)，无法放入 V100 16GB

## 替代方案

1. **使用更小的模型**: Qwen2.5-7B-Instruct、Llama-3-8B
2. **升级 GPU**: A100 40GB 或更大
3. **使用 8 张 GPU** (TP=8) 进一步分片
