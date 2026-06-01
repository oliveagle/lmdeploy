# Rust Server 性能优化 - Beads 任务总览

## 创建时间
2026-06-02

## Epic: lmdeploy-1rke
**标题:** Rust Server 超越 Python TurboMind Prefill 性能
**优先级:** P0
**类型:** epic

## 任务层级结构

```
lmdeploy-1rke (Epic: Rust Server 超越 Python TurboMind Prefill 性能)
├── lmdeploy-1rke.1 (Story 1: 修复权重加载 BUG)
│   ├── lmdeploy-1rke.1.1 (Story 2: 修复 QKV Fusion BUG)
│   ├── lmdeploy-1rke.1.2 (Story 3: 修复 Buffer 空指针 BUG)
│   └── lmdeploy-1rke.1.3 (Story 4: 验证基准性能达到 Python 水平)
└── lmdeploy-1rke.2 (Story 5: 优化 Prefill 性能超越 Python 20%+)
```

## 任务详情

### lmdeploy-1rke.1: Story 1: 修复权重加载 BUG
- **优先级:** P0
- **类型:** task
- **估算:** 120 分钟
- **标签:** rust,turbomind,bug,weight-loading
- **描述:** 修复 Rust Server 权重完全未加载的问题

### lmdeploy-1rke.1.1: Story 2: 修复 QKV Fusion BUG
- **优先级:** P0
- **类型:** task
- **估算:** 120 分钟
- **标签:** rust,turbomind,bug,qkv
- **描述:** 修复 AWQ 模型 QKV 权重加载失败

### lmdeploy-1rke.1.2: Story 3: 修复 Buffer 空指针 BUG
- **优先级:** P0
- **类型:** task
- **估算:** 60 分钟
- **标签:** rust,turbomind,bug,buffer
- **描述:** 修复 Buffer 空指针崩溃

### lmdeploy-1rke.1.3: Story 4: 验证基准性能达到 Python 水平
- **优先级:** P0
- **类型:** task
- **估算:** 60 分钟
- **标签:** rust,turbomind,benchmark,performance
- **描述:** 验证修复后的性能
- **依赖:** Story 1, Story 2, Story 3

### lmdeploy-1rke.2: Story 5: 优化 Prefill 性能超越 Python 20%+
- **优先级:** P1
- **类型:** task
- **估算:** 240 分钟
- **标签:** rust,turbomind,performance,optimization
- **描述:** 性能优化，超越 Python 20%以上

## 执行进度

### 当前状态: 全部待执行

所有任务都是新建状态 (●)，需要开始执行。

### 执行顺序

1. **Phase 1: BUG 修复 (必须)**
   - lmdeploy-1rke.1: 修复权重加载 BUG
   - lmdeploy-1rke.1.1: 修复 QKV Fusion BUG
   - lmdeploy-1rke.1.2: 修复 Buffer 空指针 BUG

2. **Phase 2: 性能验证**
   - lmdeploy-1rke.1.3: 验证基准性能达到 Python 水平

3. **Phase 3: 性能优化**
   - lmdeploy-1rke.2: 优化 Prefill 性能超越 Python 20%+

## 目标性能

| Context | Python (tok/s) | Rust 目标 (tok/s) | 提升比例 |
|---------|----------------|-------------------|----------|
| 4k      | 3915           | 4700+             | +20%     |
| 8k      | 3319           | 4000+             | +20%     |
| 16k     | 2377           | 2850+             | +20%     |

## 相关文档

- PRD: `.research/prd_rust_server_prefill_performance_20260602.md`
- 性能分析: `.research/rust_server_performance_analysis_20260602.md`
- Python 基准: `benchmark/benchmark_pipeline_4k_8k_16k.py`
- Rust 基准: `lmdeploy-rust-server/src/bin/prefill_benchmark_4_8_16k.rs`

## 下一步行动

1. 开始执行 lmdeploy-1rke.1 (修复权重加载 BUG)
2. 调试 `src/turbomind/capi/turbomind_c.cc` 权重加载逻辑
3. 找到 "Loaded 0 tensors" 的根本原因
4. 修复后继续执行后续任务

## 验证标准

**总体验收:**
- [ ] Rust Server 能够正常运行 4k/8k/16k prefill 基准测试
- [ ] Prefill 性能超越 Python TurboMind 20%以上
- [ ] Decode 性能不低于 Python
- [ ] 无崩溃，无内存泄漏
- [ ] 代码通过 review
