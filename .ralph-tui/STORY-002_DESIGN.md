# LMDeploy DFlash EP=4 支持 - 架构设计

> **创建时间**: 2026-05-10 22:00
> **最后更新**: 2026-05-10 22:00
> **相关记录**: [.ralph-tui/progress.md](.ralph-tui/progress.md)

## 概述

在 Turbomind 后端实现 Expert Parallelism (EP=4) 支持，使 Qwen3.6-35B-A3B-AWQ 模型能在 4x V100 16GB 上以 EP=4, TP=1 配置运行，作为 DFlash 性能对比的基线。

## 当前状态分析

### 已有基础

以下组件已存在但尚未完全集成：

| 组件 | 位置 | 状态 |
|------|------|------|
| `TurbomindEngineConfig.ep` | `lmdeploy/messages.py:290` | `ep: int = 1`，已定义 |
| `ModelConfig.mlp_ep_size` | `lmdeploy/turbomind/deploy/config.py:74` | `mlp_ep_size: int = 1`，已定义 |
| `ModelConfig.mlp_ep_rank` | `lmdeploy/turbomind/deploy/config.py:75` | `mlp_ep_rank: int = 0`，已定义 |
| `MoeParam.ep_size_` | `src/turbomind/models/llama/llama_params.h:117` | 已定义 |
| `MoeParam.ep_rank_` | `src/turbomind/models/llama/llama_params.h:121` | 已定义 |
| `MoeFfnLayer` EP 逻辑 | `src/turbomind/models/llama/moe_ffn_layer.cc:644-649` | EP>1 时 all_reduce 已实现 |
| PyTorch EP 参考 | `lmdeploy/pytorch/distributed.py:203-233` | `_build_ep_group` 已实现 |

### 缺失环节

1. **权重分片**: 当前 `MoeFfn.apply()` 按 TP 分片权重，未考虑 EP
2. **参数传递**: Python 端 `TurbomindEngineConfig.ep` → `ModelConfig.mlp_ep_size` **未传递到 C++**
   - `converter.py:273-278` 只设置 `attn_tp_size`, `mlp_tp_size`，**未设置 `mlp_ep_size`**
   - `turbomind.cc:542-548` 只读取 `attn_tp_size`, `mlp_tp_size`，**未读取 `mlp_ep_size`**
3. **专家分配**: C++ 层已有 `expert_assignment` 逻辑 (`LlamaDenseWeight.cc:577-587`)
4. **集合通信**: Turbomind 已有 `all_reduce`，但 EP 模式下可能需要 `alltoall`

## 架构设计

### 1. EP + TP 组合策略

```
总 GPU 数 = EP × TP
├── EP group: ep_size 个 GPU，共享全部专家但各自负责不同专家子集
└── TP group: tp_size 个 GPU，在 EP group 内张量并行
```

**本方案**: EP=4, TP=1（4 个 GPU，每个负责 1/4 专家）

### 2. 专家分配逻辑

```
总专家数: 256 (Qwen3.6-35B-A3B-AWQ)
EP=4 → 每个 GPU 负责: 256/4 = 64 个专家
expert_assignment:
  - Rank 0: experts [0, 63]
  - Rank 1: experts [64, 127]
  - Rank 2: experts [128, 191]
  - Rank 3: experts [192, 255]
```

### 3. 数据流

```
输入 → MoE Router (local)
     → All2All: 根据 expert_assignment 路由 token 到正确 EP rank
     → Local Experts (每个 rank 只计算自己负责的专家)
     → All2All: 收集结果回原始 token 顺序
     → Output
```

### 4. 需要修改的文件

#### Python 层

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `lmdeploy/messages.py` | `TurbomindEngineConfig.ep` 已存在 | ✅ |
| `lmdeploy/turbomind/deploy/config.py` | `ModelConfig.mlp_ep_size/mlp_ep_rank` 已存在 | ✅ |
| `lmdeploy/turbomind/deploy/converter.py` | **需添加**: 设置 `model_config.mlp_ep_size/mlp_ep_rank` | ❌ |
| `lmdeploy/turbomind/deploy/module.py` | **需修改**: `MoeFfn.apply()` 按 EP 分片权重 | ❌ |
| `lmdeploy/turbomind/turbomind.py` | **需添加**: 传递 EP 参数到 C++ | ❌ |

#### C++ 层

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `src/turbomind/models/llama/llama_params.h` | `MoeParam.ep_size_`/`ep_rank_` 已存在 | ✅ |
| `src/turbomind/models/llama/LlamaDenseWeight.cc` | **EP expert_assignment 逻辑已实现** | ✅ |
| `src/turbomind/models/llama/moe_ffn_layer.cc` | EP all_reduce 已实现 | ✅ |
| `src/turbomind/turbomind.cc` | **需添加**: 读取 `mlp_ep_size/mlp_ep_rank` | ❌ |
| `src/turbomind/models/llama/LlamaWeight.cc` | **已传递**: `mlp_ep_size` 到 LlamaWeight | ✅ |

### 5. 实现优先级

1. **P0**: 参数传递 (Python → C++ 参数链路打通)
   - `converter.py`: 添加 `mlp_ep_size` 设置
   - `turbomind.cc`: 添加 `mlp_ep_size` 读取
2. **P1**: 权重分片 (按 EP 分片专家权重)
   - `module.py:MoeFfn.apply()`: 按专家范围分片
3. **P2**: 集合通信 (验证 All2All/all_reduce)
   - 验证现有通信逻辑是否支持 EP
4. **P3**: 测试验证

## 关键设计决策

### 决策 1: 复用 TP 通信 vs 新建 EP group

**选择**: 复用现有 TP 通信基础，在 `ep_size > 1` 时建立 EP 特定的 NCCL group。

**原因**: 
- Turbomind 的 NCCL 初始化已相对完善
- EP group 与 TP group 在 EP=4, TP=1 时重合
- 避免过度重构

### 决策 2: 权重分片策略

**选择**: 按专家维度分片（而非按专家内权重维度分片）。

**原因**:
- EP 的核心思想是专家分片
- 简单且内存效率高
- 与 PyTorch 后端 EP 实现一致

### 决策 3: All2All 通信实现

**选择**: 使用 NCCL `alltoall` 而非自定义实现。

**原因**:
- NCCL alltoall 已优化
- 减少自定义代码量
- 与 PyTorch 后端一致

## 参数传递链路分析

### 当前链路 (TP 支持)

```
TurbomindEngineConfig.tp
  → converter.py (set attn_tp_size, mlp_tp_size)
    → TurbomindModelConfig.model_config (attn_tp_size, mlp_tp_size)
      → export_config() → config.yaml
        → turbomind.cc (read attn_tp_size, mlp_tp_size)
          → EngineParam (attn_tp_size, mlp_tp_size)
            → LlamaWeight (tp_size_, tp_rank_)
```

### 需要添加的链路 (EP 支持)

```
TurbomindEngineConfig.ep
  → converter.py (❌ 缺失: set mlp_ep_size)
    → TurbomindModelConfig.model_config (mlp_ep_size) ✅ 已定义
      → export_config() → config.yaml ✅ 已导出
        → turbomind.cc (❌ 缺失: read mlp_ep_size)
          → EngineParam (mlp_ep_size) ✅ 已定义
            → LlamaWeight (mlp_ep_size_) ✅ 已使用
```

### 关键发现

1. **C++ 层已实现 EP 逻辑**: `LlamaDenseWeight.cc:577-587` 已有 `expert_assignment` 计算
2. **参数传递是瓶颈**: Python → C++ 的 `mlp_ep_size` 参数传递链路不完整
3. **权重分片未实现**: `MoeFfn.apply()` 未按 EP 分片专家

- PyTorch EP 实现: `lmdeploy/pytorch/distributed.py`
- MoE 实现参考: `src/turbomind/models/llama/moe_ffn_layer.cc`
- 配置定义: `lmdeploy/turbomind/deploy/config.py`

---

*Design Doc v1.0 - 2026-05-10 22:00*
