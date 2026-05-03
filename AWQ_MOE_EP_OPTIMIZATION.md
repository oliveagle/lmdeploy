# AWQ MoE EP 性能优化方案

> **目标**: 提升 4x V100 EP=4 配置下的 Qwen3.6-35B-A3B-AWQ 性能

## 问题分析

### 当前瓶颈

1. **AWQ 解量化开销** - 每次前向都解量化，无缓存
2. **专家通信开销** - EP token dispatch/combine
3. **内存带宽限制** - V100 HBM2 带利不足

### 性能数据

| 配置 | Decode 吞吐 | 问题 |
|------|-------------|------|
| 单卡小模型 | ~12 tokens/s | 基准 |
| EP=4 (预期) | ~40+ tokens/s | 线性扩展未达到 |

## 优化方案

### 方案 1: 权重预解量化缓存 ⭐ 推荐

**原理**: 在模型加载时预解量化所有专家权重，缓存为 fp16

**优点**:
- 消除运行时解量化开销
- 简单实现，风险低
- 预期提升: 2-3x

**缺点**:
- 内存占用增加 (int4 → fp16)
- EP=4 下每卡约需: 15GB × 2 = 30GB (超出 16GB)

**实现**:
```python
# 在 FusedMoEAWQ.__init__ 中添加:
if ENABLE_WEIGHT_CACHE:
    self.cached_gate_up = self._pre_dequantize_weights()
    self.cached_down = self._pre_dequantize_down_weights()
```

**适用场景**: 大显存 GPU (A100 40GB+)

---

### 方案 2: 分层预解量化

**原理**: 只预解量化高频专家，动态解量化低频专家

**优点**:
- 平衡内存和计算
- 适合 16GB 显存

**实现**:
```python
# 统计专家使用频率
expert_freq = self._compute_expert_frequency()
# 预解量化 top 50% 专家
hot_experts = sorted(expert_freq.items(), key=lambda x: -x[1])[:num_experts//2]
self._pre_dequantize_partial(hot_experts)
```

**预期提升**: 1.5-2x

---

### 方案 3: 优化 fused_moe kernel

**原理**: 修改 Triton kernel 直接支持 int4 权重

**优点**:
- 无额外内存
- 理论最优性能

**缺点**:
- 开发工作量大
- 需要深入修改 kernel

**实现位置**: `lmdeploy/pytorch/kernels/cuda/fused_moe.py`

---

### 方案 4: 减少 EP 通信开销

**原理**: 优化 token dispatcher 和 combine 操作

**优化点**:
1. 使用 NCCL instead of gloo
2. 合并 dispatch/combine 操作
3. 使用 in-place 操作减少内存拷贝

**实现位置**:
- `lmdeploy/pytorch/backends/cuda/moe/default.py`
- `lmdeploy/pytorch/backends/cuda/moe/ep_utils.py`

---

### 方案 5: 混合精度推理

**原理**: 关键路径用 fp16，非关键用 fp32

**优点**:
- 利用 Tensor Core
- 减少 HBM 压力

**实现**:
```python
# 解量化用 fp32，计算用 fp16
weights_fp32 = self._dequantize(...)
weights_fp16 = weights_fp32.to(torch.float16)
```

---

## 推荐实施路线

### 阶段 1: 快速优化 (1-2 天)

1. **启用 NCCL backend** (如果未启用)
2. **优化解量化**: 添加权重缓存选项
3. **调整 batch size**: 测试最优 batch

### 阶段 2: 中期优化 (3-5 天)

1. **实现分层预解量化**
2. **优化 EP 通信路径**
3. **添加性能分析工具**

### 阶段 3: 深度优化 (1-2 周)

1. **开发 int4 fused_moe kernel**
2. **实现 expert scheduling**
3. **CUDA Graph 优化**

## 配置建议

### V100 16GB x4 最优配置

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
    # 新增优化参数
    enable_weight_cache=False,  # 显存不足
    enable_partial_cache=True,  # 分层缓存
    cache_hot_experts=64,       # 缓存 64 个热点专家
    use_nccl=True,              # 使用 NCCL
)
```

### A100 40GB x4 最优配置

```python
PytorchEngineConfig(
    # ... 同上 ...
    enable_weight_cache=True,   # 全量缓存
)
```

## 性能预期

| 优化方案 | 预期提升 | 实施难度 |
|----------|----------|----------|
| 权重预解量化 | 2-3x | 低 |
| 分层预解量化 | 1.5-2x | 中 |
| 优化 EP 通信 | 1.2-1.5x | 中 |
| int4 kernel | 2-4x | 高 |
| 组合优化 | 3-5x | - |

## 监控指标

1. **Decode 吞吐**: tokens/s
2. **GPU 利用率**: `nvidia-smi`
3. **内存占用**: `torch.cuda.memory_allocated()`
4. **解量化时间**: 添加 profiler

## 相关文件

- `lmdeploy/pytorch/nn/moe/awq.py` - AWQ MoE 实现
- `lmdeploy/pytorch/backends/cuda/moe/default.py` - EP backend
- `lmdeploy/pytorch/kernels/cuda/awq_kernels.py` - AWQ kernels
- `lmdeploy/pytorch/kernels/cuda/fused_moe.py` - Fused MoE kernel
