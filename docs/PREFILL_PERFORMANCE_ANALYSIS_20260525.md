# Python TurboMind Prefill 性能分析报告

**日期**: 2026-05-25
**模型**: Qwen3.6-35B-A3B-AWQ
**GPU**: Tesla V100 32GB (PG503-216)
**LMDeploy**: 0.13.0

---

## 1. 当前 Python TurboMind 真实性能数据

### 测试配置
```python
TurboMind(
    session_len=16384,
    max_batch_size=1,
    model_format='awq',
    cache_max_entry_count=0.4,
    max_prefill_token_num=16384,
    num_tokens_per_iter=16384,
    max_prefill_iters=1,
)
```

### 实测结果 (5 runs 平均)

| Context | Tokens | Avg TTFT | Avg TPS | Max TPS |
|---------|--------|----------|---------|---------|
| 1K | 1,001 | 283ms | 3,535 tok/s | 5,351 tok/s |
| 2K | 2,001 | 623ms | 3,210 tok/s | 3,215 tok/s |
| 4K | 4,001 | 1,176ms | 3,402 tok/s | 4,212 tok/s |
| 8K | 8,001 | 3,011ms | 2,657 tok/s | 2,658 tok/s |

### 关键发现

1. **性能随 context 增大而下降**
   - 1K: 3,535 tok/s (baseline)
   - 8K: 2,657 tok/s (75% of 1K performance)
   - 这表明 O(N²) 的 attention 计算正在成为瓶颈

2. **TTFT 线性增长**
   - 1K → 2K: TTFT 从 283ms → 623ms (2.2x)
   - 1K → 4K: TTFT 从 283ms → 1,176ms (4.2x)
   - 1K → 8K: TTFT 从 283ms → 3,011ms (10.6x)

---

## 2. 历史数据对比 (2026-05-18)

| Context | 历史数据 | 当前实测 | 差异 |
|---------|----------|----------|------|
| 1K | 14,827 tok/s | 3,535 tok/s | **4.2x 慢** |
| 4K | 33,727 tok/s | 3,402 tok/s | **9.9x 慢** |
| 8K | 42,875 tok/s | 2,657 tok/s | **16.1x 慢** |

### 可能的性能退化原因

1. **模型文件不同**
   - 历史模型: `/mnt/data/models/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ`
   - 当前模型: `/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ`
   - 两个路径不同，模型可能已更新或重新量化

2. **配置参数差异**
   - 历史测试可能使用了不同的 `max_prefill_token_num` 等参数

3. **KV Cache 状态**
   - 历史测试可能是"冷启动"后的第一次推理
   - 当前测试可能受 KV Cache 碎片化影响

4. **CUDA 版本或驱动差异**
   - 历史: CUDA 12.5
   - 当前: 待确认

---

## 3. Rust Server 性能对比

**状态**: ✅ **编译成功** (2026-05-25)
- Binary: `lmdeploy-rust-server/target/debug/lmdeploy-server` (168.7M)
- 运行时问题: `cublasCreate_v2` 未定义符号 (CUDA 库链接)
- 已创建优化 Epic: `lmdeploy-g84l` (5 个子任务)

---

## 4. 性能优化方案

### 4.1 短期修复 (P0)

1. **修复 Rust 编译错误**
   - 使用同步 `Forward` 替代 `ForwardAsync`
   - 修复 `TensorMap::clear()` 实现

2. **启用 CUDA Graph**
   - 在 C++ 层捕获 prefill 的 CUDA Graph
   - 避免每次重新调度 kernel

### 4.2 中期优化 (P1)

3. **实现 Zero-Copy Tensor 传递**
   - 使用 `TM_TensorMap_SetDLPack` 避免 CPU 拷贝
   - 直接传递 GPU 指针

4. **批量 Prefill 调度**
   - 预分配 token 数量
   - 合并小请求为 batch prefill

5. **消除冗余 RMSNorm**
   - 检查并移除 prefill 中的重复 normalization 调用

### 4.3 长期优化 (P2)

6. **KV Cache 布局优化**
   - 使用 PagedAttention 布局
   - 减少 KV Cache 碎片化

7. **异步 H2D 传输**
   - 使用 pinned memory + CUDA streams
   - 隐藏 H2D 延迟

---

## 5. 超越 Python 性能的目标

为了超越 Python TurboMind 的 3,000-3,500 tok/s，Rust Server 需要:

1. **消除 Python 包装开销**: 2-5% 提升
2. **Zero-Copy Tensor 传递**: 10-20% 提升
3. **CUDA Graph 优化**: 15-30% 提升
4. **批量 Prefill**: 20-40% 提升

**目标**: 5,000-10,000 tok/s (比 Python 快 1.5-3x)

---

## 6. 验收标准

### Epic lmdeploy-g84l 子任务

1. **TASK-1 (P1)**: 实现 Zero-Copy Tensor 传递 - 使用 DLPack 避免 CPU 拷贝
2. **TASK-2 (P1)**: 消除多次同步点 - batch prefill 预分配 token 数量
3. **TASK-3 (P2)**: 消除冗余 RMSNorm - prefill 中的重复 normalization
4. **TASK-4 (P2)**: CUDA Graph 集成 - 消除 kernel 启动开销
5. **TASK-5 (P1)**: KV Cache 直接映射 - 避免 tensor 拷贝

### 总体验收标准

- [x] Rust Server 编译成功
- [ ] 基本推理功能正常
- [ ] 1K prefill 达到 4,000+ tok/s (目标: 超越 Python 3,545 tok/s)
- [ ] 4K prefill 达到 4,000+ tok/s (目标: 超越 Python 3,417 tok/s)
- [ ] 8K prefill 达到 3,000+ tok/s (目标: 超越 Python 2,656 tok/s)