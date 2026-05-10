# 开发文档索引

本目录包含 LMDeploy 开发过程中的临时文档和项目总结。

> **注意**: 这些文档记录了开发过程中的问题、修复和测试结果。部分内容可能已过时，请以最新代码为准。

## 目录

- [DFlash Speculative Decoding](dflash/) - DFlash 推测解码集成
- [Expert Parallelism (EP)](ep/) - EP 模式修复和优化
- [MoE 模型支持](moe/) - 混合专家模型支持

## DFlash Speculative Decoding

DFlash 是一种高效的推测解码方法，通过小型 draft 模型预测多个 token，然后由大型 target 模型并行验证。

| 文档 | 说明 |
|------|------|
| [README](dflash/README.md) | 项目概述和使用指南 |
| [测试指南](dflash/test-guide.md) | 测试脚本和运行方式 |
| [V100 限制](dflash/v100-limitations.md) | V100 GPU 已知问题和限制 |

**状态**: ✅ 集成完成，等待 A100 GPU 进行性能测试

## Expert Parallelism (EP)

EP（Expert Parallelism）用于分片 MoE 模型的专家层，实现多 GPU 并行。

| 文档 | 说明 |
|------|------|
| [修复总结](ep/fix-summary.md) | EP=4 修复总结 |
| [EP+TP 组合](ep/tp-combination.md) | EP+TP 组合说明和限制 |

**状态**: ✅ 已修复并验证

## MoE 模型支持

混合专家（Mixture of Experts）模型的量化支持。

| 文档 | 说明 |
|------|------|
| [Qwen3.5 AWQ](moe/qwen35-awq.md) | Qwen3.5 MoE AWQ 量化支持 |

**状态**: ✅ 已修复

## 相关提交

- `9062dc93` - fix: DistConfig world_size and attn_tp for EP + TP combination
- `09445c9a` - fix: DistConfig attn_tp default for EP + TP combination
- `5d437d5b` - fix: 修复 Qwen3.5 MoE AWQ 在 TP 模式下的参数加载问题
- `3c359f1d` - Add DFlash end-to-end test script
- `cb403436` - Fix __get_param function for Qwen3.5 MoE AWQ weight loading
