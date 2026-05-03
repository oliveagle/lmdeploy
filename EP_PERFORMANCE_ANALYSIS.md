# EP=4 性能问题根因分析与优化方案

## 当前性能数据 (4x V100 16GB)

### 测试结果分析

| 指标 | 数值 | 问题 |
|------|------|------|
| Decode 吞吐 | ~46 tokens/s | 远低于预期 |
| Prefill 延迟 | ~16.5s | 极慢 |
| 内存占用 | ~1.6 GB/卡 | 正常 |

### 预期 vs 实际

| 配置 | 预期 Decode | 实际 Decode | 差距 |
|------|-------------|-------------|------|
| 单卡 | ~12 tok/s | ~12 tok/s | ✅ 基准正常 |
| 4卡 EP | ~48 tok/s | ~46 tok/s | ❌ 几乎无提升 |

## 🔍 根因分析

### 1. EP 通信开销过大 ⭐ 主要瓶颈

**问题**:
- 当前使用 `torch.distributed` 进行 all_gather/all_reduce
- 通信同步开销大，掩盖了计算优势
- Token dispatch/combine 未优化

**证据**:
```python
# 从 awq.py:238-242 可以看到
hidden_states, topk_weights, topk_idx = moe_gather_inputs(
    state['hidden_states'],
    state['topk_weights'],
    state['topk_idx'],
    group=self.gather_group  # ← 这里导致同步等待
)
```

### 2. AWQ 解量化在每次调用时执行

**问题**:
- 每层 MoE 都要解量化权重
- 无缓存机制
- CPU-GPU 数据传输

**证据**:
```python
# awq.py:298
gate_up_weights, down_weights = self._get_weights([...])  # 每次都解量化
```

### 3. Triton Kernel 配置未针对 V100 优化

**问题**:
- V100 (SM 70) 与 A100 (SM 80) 的最优配置不同
- 当前 autotune 配置可能选择了不适合 V100 的参数

### 4. 专家负载不均衡

**问题**:
- 随机 topk_ids 导致某些 GPU 空闲
- 未实现 dynamic load balancing

## 🎯 优化方案 (按优先级)

### 优先级 1: 减少 EP 通信开销

**方案 A: 使用 NCCL backend**
```bash
# 当前可能使用 gloo，切换到 nccl
torch.distributed.init_process_group(backend='nccl')
```

**预期提升**: 1.3-1.5x

**方案 B: 异步通信**
```python
# 在 gemm 前就开始通信
# 使用 torch.distributed.isend/irecv
```

**预期提升**: 1.2-1.4x

### 优先级 2: 权重预解量化

**方案**: 在模型加载时解量化所有专家

**限制**:
- V100 16GB 显存不足
- 需要使用 partial cache (只缓存热点专家)

**实施**:
```python
# 缓存 32 个热点专家 (128 total / 4 = 32 per rank)
moe = FusedMoEAWQ(
    ...
    enable_weight_cache=True,
    cache_hot_experts=32,
)
```

**预期提升**: 1.5-2x

### 优先级 3: 优化 Triton Kernel

**方案**: 针对 V100 (SM 70) 调整配置

**位置**: `lmdeploy/pytorch/kernels/cuda/fused_moe.py:12-80`

**优化**:
```python
# V100 最优配置
V100_CONFIGS = [
    triton.Config({
        'BLOCK_SIZE_M': 128,
        'BLOCK_SIZE_N': 128,
        'BLOCK_SIZE_K': 32,
    }, num_stages=4, num_warps=4),
    # ... 更多 V100 专用配置
]
```

**预期提升**: 1.1-1.3x

### 优先级 4: 专家调度优化

**方案**: 使用 TopK 路由缓存

**原理**: 相邻 token 通常选择相同的专家

**实施**:
```python
# 缓存最近的路由决策
if hasattr(self, '_last_topk_ids') and torch.all(topk_ids == self._last_topk_ids):
    # 复用上次的调度结果
    pass
self._last_topk_ids = topk_ids.clone()
```

**预期提升**: 1.1-1.2x

## 📊 综合优化预期

| 优化组合 | 预期 Decode 吞吐 | 提升倍数 |
|----------|------------------|----------|
| 当前 (基线) | 46 tok/s | 1.0x |
| + NCCL | 60 tok/s | 1.3x |
| + 权重缓存 (32) | 90 tok/s | 1.95x |
| + Kernel 优化 | 100 tok/s | 2.17x |
| + 专家调度 | 110 tok/s | 2.39x |

## 🚀 快速实施步骤

### 步骤 1: 确认当前 backend

```bash
# 检查是否使用 NCCL
python3 -c "import torch; print(torch.distributed.is_nccl_available())"
```

### 步骤 2: 运行优化测试

```bash
# 测试 NCCL 优化
torchrun --nproc_per_node=4 test_awq_moe_ep_nccl.py

# 测试权重缓存
torchrun --nproc_per_node=4 test_awq_moe_ep_cache.py

# 测试组合优化
torchrun --nproc_per_node=4 test_awq_moe_ep_combined.py
```

### 步骤 3: 性能分析

```bash
# 使用 Nsight Systems 分析
nsys profile -o profile_report python3 test_script.py

# 查看热点
nsys stats profile_report.nsys-rep
```

## 🔧 关键修改文件

1. **lmdeploy/pytorch/nn/moe/awq.py** - 权重缓存
2. **lmdeploy/pytorch/backends/cuda/moe/default.py** - EP backend
3. **lmdeploy/pytorch/kernels/cuda/fused_moe.py** - Kernel 配置
4. **lmdeploy/pytorch/kernels/cuda/awq_kernels.py** - AWQ 解量化

## ⚠️ V100 特定注意事项

1. **显存限制**: 16GB 不足以缓存所有专家
2. **Compute Capability 7.0**: 需要专用 kernel 配置
3. **HBM2 带宽**: 900 GB/s，比 A100 的 2 TB/s 慢很多
4. **Tensor Core**: V100 只有 FP16 Tensor Core，无 BF16

## 📈 监控指标

优化过程中需要监控：

1. **GPU 利用率** (`nvidia-smi dmon -s u`)
2. **内存带宽** (`ncu --metrics dram__throughput.avg.pct_of_peak`)
3. **通信时间** (添加 profiler)
4. **解量化时间** (添加 timer)
