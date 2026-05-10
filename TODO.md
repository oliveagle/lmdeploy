# LMDeploy + DFlash TODO

> **最后更新**: 2026-05-10 22:15
> **当前状态**: 实现 Turbomind EP=4 支持 - 阶段1架构设计已完成

## 最新进展 🔥

- **2026-05-10 22:15**: STORY-002 架构设计完成，详细设计文档已创建
- **2026-05-10 22:00**: 开始实现 Turbomind EP=4 支持
- **2026-05-10 19:10**: DFlash 集成验证通过 (所有测试 ✓)
- **2026-05-10 19:00**: 优化 draft token 数量 (8 → 16，可配置)
- **2026-05-10 18:45**: Python 集成验证完成 (`turbomind.py` 已支持 DFlash)
- **2026-05-10 18:30**: 修改为**非因果 attention** (关键优化!)
- **2026-05-10 18:15**: Turbomind 编译成功，DFlash C++ 集成完成
- **2026-05-10 18:00**: 分析 lucebox-hub/dflash 参考实现

## 已完成 ✅

- [x] DFlash C++ 源文件创建 (DFlashDraftModel, DFlashDraftWeight, dflash_kernels)
- [x] Turbomind 集成 (LlamaWeight, LlamaDecoder, unified_decoder)
- [x] CMakeLists.txt 编译配置
- [x] Turbomind 编译成功 (`import lmdeploy.lib._turbomind` OK)
- [x] 分析 lucebox-hub/dflash 参考实现
- [x] **修改为非因果 attention** (2026-05-10 18:30)
- [x] **Python 集成验证** (2026-05-10 18:45)
- [x] **优化 draft token 数量 (8 → 16，可配置)** (2026-05-10 19:00)
- [x] **优化 aux layers 选择 (验证当前值已是最优)** (2026-05-10 19:05)
- [x] **DFlash 集成验证测试** (2026-05-10 19:10)

## Turbomind EP=4 支持计划 🚀 (2026-05-10 22:00)

### 背景
用户需要 Turbomind 后端支持 EP=4, TP=1, KV TurboQuant 配置，以便：
1. 作为 DFlash 性能对比的基线
2. 在 4x V100 16GB 上运行 Qwen3.6-35B-A3B-AWQ 模型
3. 输出文本质量必须正常

### 当前状态
- **Turbomind**: C++ 层已有部分 EP 支持（expert_assignment、all_reduce），但参数传递链路不完整
- **PyTorch 后端**: 支持 EP=4，但输出异常 (全感叹号)
- **设计文档**: `.ralph-tui/STORY-002_DESIGN.md` 已创建

### 实现计划

#### 阶段 1: 架构设计
- [x] 分析 Turbomind MoE 实现 (moe_ffn_layer.cc, llama_params.h)
- [x] 分析 PyTorch 后端 EP 实现参考
- [x] 设计 Turbomind EP 支持架构 (STORY-002_DESIGN.md)
- [x] 定义 EP 配置参数 (ep_size, ep_rank, expert_assignment)

#### 阶段 2: C++ 核心实现
- [ ] 修改 `EngineParam` 添加 EP 参数
- [ ] 修改 `MoeFfnLayer` 支持 EP
- [ ] 实现专家分片逻辑 (expert_partition.h/cc)
- [ ] 修改 MoE 权重加载支持 EP
- [ ] 实现 EP 集合通信 (allgather/reduce_scatter)

#### 阶段 3: Python 集成
- [ ] 修改 `TurbomindEngineConfig` 添加 EP 参数
- [ ] 修改 Turbomind 部署转换器支持 EP
- [ ] 修改模型权重加载支持 EP
- [ ] 修改并行配置初始化

#### 阶段 4: 测试验证
- [ ] 创建 EP=4 测试脚本
- [ ] 验证模型加载正常
- [ ] 验证输出质量正常
- [ ] 性能基准测试
- [ ] 与 DFlash 性能对比

#### 阶段 5: 文档更新
- [ ] 更新 EP 使用文档
- [ ] 记录配置参数
- [ ] 记录性能指标

### 关键文件
**C++ 核心**:
- `src/turbomind/models/llama/llama_params.h` - EngineParam, MoeParam 定义
- `src/turbomind/models/llama/moe_ffn_layer.h` - MoE FFN 层
- `src/turbomind/models/llama/moe_ffn_layer.cc` - MoE FFN 实现

**Python 配置**:
- `lmdeploy/messages.py` - TurbomindEngineConfig, QuantPolicy
- `lmdeploy/turbomind/deploy/config.py` - ModelConfig 定义
- `lmdeploy/turbomind/deploy/parameter.py` - 参数字典

**权重转换**:
- `lmdeploy/turbomind/deploy/source_model/qwen.py` - Qwen3_5MoeModel

**参考**:
- `lmdeploy/pytorch/distributed.py` - PyTorch EP 实现
- `lmdeploy/pytorch/nn/moe/awq.py` - PyTorch AWQ MoE EP

### 关键配置
```
EP=4, TP=1
KV Cache: TurboQuant (K=4bit, V=2bit)
Session Length: 2048
Max Batch Size: 1
```

## 进行中 🚧

### 优先级 0: Turbomind EP=4 支持 🔥

#### 当前任务
- [x] 设计 Turbomind EP 架构 (2026-05-10 22:15)
- [ ] 实现 C++ 核心 EP 逻辑
- [ ] 实现 Python 集成
- [ ] 测试验证输出质量

### 待办任务 📋

### 优先级 1: 功能完善

#### 1.1 Python 集成
- [x] 添加 `SpeculativeConfig` 到 `lmdeploy/messages.py` (已存在)
- [x] 修改 `turbomind.py` Python 接口支持 DFlash (已完成)
- [x] 添加 DFlash draft model 权重加载逻辑 (已完成)

#### 1.2 内存优化
- [x] KV cache 内存使用 (已有 TurboQuant, INT4, INT8 支持)
- [x] Draft 模型使用 Q4 量化 (STORY-008 已完成)
- [ ] 考虑 CPU embedding (不占用 GPU)

#### 1.3 测试验证
- [x] DFlash 集成验证测试 (2026-05-10 19:10)
- [ ] 在 A100 40GB 上测试
- [ ] 或使用 8x V100 (TP=8)
- [ ] 或测试更小的模型 (Qwen3.5-14B)

### 优先级 2: 性能优化

#### 2.1 Attention 优化
- [x] 非 causal attention (已完成)
- [ ] Flash Attention 滑动窗口
- [ ] RoPE 优化 (NEOX style, theta=10M)

#### 2.2 Verification 优化
- [ ] DDTree verification (tree-structured verify)
- [ ] 优化 verify kernel 性能

#### 2.3 Draft 优化
- [x] 非 causal attention (已完成 2026-05-10 18:30)
- [x] 优化 draft token 数量 (8 → 16，可配置) (已完成 2026-05-10 19:00)
- [x] 优化 aux layers 选择 (当前 {1,10,19,28,37} 已是 40 层模型的最优值) (已完成 2026-05-10 19:05)
- [x] **DFlash 集成验证测试** (已完成 2026-05-10 19:10)

### 优先级 3: 高级功能

#### 3.1 长文本支持
- [ ] 支持 128K context (TQ3_0 KV 量化)
- [ ] Prefix caching
- [ ] 跨请求共享

#### 3.2 混合架构
- [ ] 支持 Full Attention + Gated DeltaNet
- [ ] 每 4 层一个 Full Attention block

## 参考资料

- lucebox-hub/dflash: `/home/oliveagle/repos/github.com/others/lucebox-hub/dflash`
- DFlash 论文: https://arxiv.org/abs/2602.06036
- DDTree 论文: https://arxiv.org/abs/2604.12989

## 硬件需求

**最小配置**:
- 4x V100 16GB (TP=4) - 当前测试失败，内存不足
- 8x V100 16GB (TP=8) - 理论上可行

**推荐配置**:
- 4x A100 40GB (TP=4) - 最佳
- 1x RTX 3090 24GB - 单卡测试

**内存需求**:
- Qwen3.6-35B-AWQ: ~18 GB/GPU (TP=4)
- Qwen3.5-27B-AWQ: ~14 GB/GPU (TP=4)
- Draft model: ~2 GB (Q4 量化后)
