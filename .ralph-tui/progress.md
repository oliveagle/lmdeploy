# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### 参数传递链路模式 (Turbomind)

Python CLI → C++ 引擎参数的标准链路：

```
TurbomindEngineConfig
  → converter.py (set model_config fields)
    → TurbomindModelConfig (export to config.yaml)
      → turbomind.cc (read from config dict)
        → EngineParam (C++ struct)
          → Model classes (LlamaWeight, etc.)
```

**关键点**:
- `converter.py` 负责将 `engine_config` 复制到 `model_config`
- `turbomind.cc` 从 YAML 解析的 dict 中读取参数
- 修改时需要同时添加 Python 端设置和 C++ 端读取

### MoE 权重分片模式

**TP 模式**: `save_split(tensor, name, split_dim=-1, split_num=tp_size)`
- 按输出维度分片权重
- 所有 rank 都存储所有专家，但专家内权重分片

**EP 模式**: (目标)
- 按 **专家维度** 分片（不同 rank 存储不同专家）
- 每个 rank 只负责 `num_experts / ep_size` 个专家
- 需要 `expert_assignment` 逻辑计算专家范围

### EP=4 Support - 已有基础

以下组件在代码库中已定义但未完全集成：

| 组件 | 位置 | 当前值 |
|------|------|--------|
| `TurbomindEngineConfig.ep` | `lmdeploy/messages.py:290` | `ep: int = 1` 已存在 |
| `ModelConfig.mlp_ep_size` | `lmdeploy/turbomind/deploy/config.py:74` | `mlp_ep_size: int = 1` 已存在 |
| `ModelConfig.mlp_ep_rank` | `lmdeploy/turbomind/deploy/config.py:75` | `mlp_ep_rank: int = 0` 已存在 |
| `MoeParam.ep_size_` | `src/turbomind/models/llama/llama_params.h:117` | `int ep_size_;` 已存在 |
| `MoeParam.ep_rank_` | `src/turbomind/models/llama/llama_params.h:121` | `int ep_rank_;` 已存在 |
| `MoeFfnLayer` EP 逻辑 | `src/turbomind/models/llama/moe_ffn_layer.cc:644-649` | 已有 all_reduce |
| PyTorch EP 参考 | `lmdeploy/pytorch/distributed.py:203-233` | `_build_ep_group` 已实现 |

### Turbomind 配置传递链路

```
TurbomindEngineConfig (Python CLI)
  → turbomind.py (create_engine())
    → ModelConfig (deploy/config.py)
      → save_model_config() → config.yaml
        → C++ EngineParam/MoeParam (llama_params.h)
```

### MoE 权重分片逻辑

**当前**: `turbomind/deploy/module.py:MoeFfn.apply()` 按 TP 分片，循环遍历所有专家
**目标**: 按 EP 分片，每个 rank 只加载自己负责的专家子集

---

## 2026-05-10 - STORY-002: 架构设计
- 完成 Turbomind EP 支持架构设计
- 创建详细设计文档: `.ralph-tui/STORY-002_DESIGN.md`
- 分析现有代码中已定义的 EP 组件和缺失环节
- **Learnings:**
  - 代码库已有大量 EP 基础设施（参数定义、部分通信逻辑），主要缺失权重分片和参数传递链路
  - `MoeFfn.apply()` 当前只考虑 TP 分片，需要修改为 EP 感知分片
  - EP group 在 EP=4, TP=1 时与 TP group 重合，可复用现有 NCCL 通信
  - 权重分片应按专家维度（而非专家内权重维度）进行
  - **关键发现**: `LlamaDenseWeight.cc:577-587` 已实现完整的 expert_assignment 逻辑，包括 `ep_first_expert_` 和 `ep_num_experts_` 计算
  - **关键发现**: `converter.py:273-278` 只传递 attn_tp_size/mlp_tp_size，缺失 mlp_ep_size 传递
  - **关键发现**: `turbomind.cc:542-548` 只读取 attn_tp_size/mlp_tp_size，缺失 mlp_ep_size 读取
  - **关键发现**: `MoeFfn.apply()` 循环遍历所有专家 (line 202)，未按 EP 分片专家范围

---
