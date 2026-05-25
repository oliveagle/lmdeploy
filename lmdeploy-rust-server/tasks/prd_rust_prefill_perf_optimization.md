# PRD: Rust Server Prefill 性能优化

## 目标

让 Rust server 的 prefill 性能 **超过** Python TurboMind 的 prefill 性能。

## 基准数据 (Python TurboMind - Qwen3.6-35B-A3B-AWQ)

两次测量取平均值：

| Context | 平均 TPS | 最大 TPS | 平均 TTFT (ms) |
|---------|---------|---------|---------------|
| 1K      | ~3,500  | ~5,300  | ~283          |
| 2K      | ~3,210  | ~3,210  | ~622          |
| 4K      | ~3,400  | ~4,200  | ~1,170        |
| 8K      | ~2,650  | ~2,650  | ~3,006        |

## 性能瓶颈分析

### 1. KV Cache 拷贝 vs 映射

**问题**: Rust FFI 层需要拷贝 KV Cache 数据，而 Python 可以直接使用 DLPack 或零拷贝映射。

**影响**: 大 batch prefill 时，数据拷贝成为显著瓶颈（O(n) 内存带宽瓶颈）。

**解决方案**: 实现 DLPack 零拷贝路径，让 Rust 直接映射 C++ 的 KV Cache 内存区域。

### 2. 多次同步点阻塞 GPU Pipeline

**问题**: Rust async/await 模型引入多次 `await` 同步点，阻塞了 GPU 执行流。

**影响**: GPU 无法充分利用并发执行，pipeline bubbles 增加。

**解决方案**: 减少 async 同步点，使用事件驱动的 CompletionCallback 替代轮询。

### 3. Batch 调度没有提前分配 Token 数量

**问题**: 当前 batch 调度是动态的，没有在 prefill 开始时就分配好所有 token 数量。

**影响**: GPU kernel launch 次数增加，无法充分利用大 batch 的并行性。

**解决方案**: 实现静态 token 分配，prefill 前一次性分配所有需要的 KV cache slots。

### 4. Prefill 中冗余的 RMSNorm 调用

**问题**: 在 prefill 流程中，Rust 层可能对每个 token 单独调用 RMSNorm，而 Python 层可以 batch 处理。

**影响**: 小 token batch 时 overhead 显著。

**解决方案**: 将 RMSNorm 调用合并到 batch 级别，减少 kernel launch overhead。

### 5. CUDA Graph 优化缺失

**问题**: Python TurboMind 使用 CUDA Graph 来优化 kernel launch，而 Rust server 没有。

**影响**: 每个 kernel launch 都有 CPU-side overhead，累计影响显著。

**解决方案**: 集成 CUDA Graph 支持，捕获 prefill 的完整执行图并重放。

## 目标性能

| Context | Python TPS | Rust 目标 TPS | 提升幅度 |
|---------|-----------|--------------|---------|
| 1K      | 3,500     | 4,500+       | +28%    |
| 2K      | 3,210     | 4,000+       | +25%    |
| 4K      | 3,400     | 4,200+       | +24%    |
| 8K      | 2,650     | 3,200+       | +21%    |

## Quality Gates

- `cargo build` 必须通过
- `cargo test` 必须通过
- prefill benchmark 必须测量并记录结果
- 每次优化后必须对比 Rust vs Python 性能

## User Stories

### US-001: 创建性能基准测量脚本

**描述**: 创建精确的 Rust server prefill benchmark 脚本，与 Python benchmark 使用相同的配置和测量方法。

**验收标准**:
- [ ] 创建与 Python benchmark 对齐的 Rust benchmark 脚本
- [ ] 支持 1K/2K/4K/8K context 测量
- [ ] 输出 JSON 格式结果，包含 avg_tps, max_tps, avg_ttft
- [ ] 能够与 Python benchmark 结果直接对比
- [ ] cargo build 通过

### US-002: 消除 KV Cache 拷贝 - 实现零拷贝路径

**描述**: 通过 DLPack 或内存映射，让 Rust 直接访问 C++ 的 KV Cache，避免数据拷贝。

**验收标准**:
- [ ] 实现 DLPack 零拷贝路径
- [ ] prefill 场景下减少内存拷贝
- [ ] cargo build 通过，无 regressions

### US-003: 减少 async 同步点 - 事件驱动完成通知

**描述**: 使用 CompletionCallback 替代轮询模式，减少 async/await 引入的同步开销。

**验收标准**:
- [ ] 实现 CompletionCallback 机制
- [ ] 移除或减少轮询等待
- [ ] 测量同步点减少对性能的影响
- [ ] cargo build 通过

### US-004: 静态 Token 分配 - Prefill 前分配所有 KV Cache Slots

**描述**: 在 prefill 开始前一次性分配所有需要的 KV cache slots，避免动态分配开销。

**验收标准**:
- [ ] 实现静态 token 数量预估
- [ ] 一次性分配 KV cache slots
- [ ] 减少动态分配的 overhead
- [ ] cargo build 通过

### US-005: 批量 RMSNorm 调用 - 减少 kernel launch

**描述**: 将多个 RMSNorm 调用合并为单次 batch 调用，减少 kernel launch overhead。

**验收标准**:
- [ ] 实现 batch RMSNorm 调用
- [ ] 减少 prefill 中的 kernel launch 次数
- [ ] cargo build 通过

### US-006: 集成 CUDA Graph 支持

**描述**: 为 prefill 流程集成 CUDA Graph 支持，捕获执行图并重放以减少 kernel launch overhead。

**验收标准**:
- [ ] 实现 CUDA Graph 捕获和重放
- [ ] prefill 场景使用 CUDA Graph 执行
- [ ] cargo build 通过
- [ ] 测量 CUDA Graph 带来的性能提升

### US-007: 综合性能验证 - Rust 超过 Python

**描述**: 在所有优化完成后，运行完整的 benchmark 对比，验证 Rust 性能超过 Python。

**验收标准**:
- [ ] Rust prefill TPS 在所有 context 长度下超过 Python
- [ ] 1K: >4,500 tok/s
- [ ] 2K: >4,000 tok/s
- [ ] 4K: >4,200 tok/s
- [ ] 8K: >3,200 tok/s
- [ ] 结果保存为 JSON 并与 Python 数据对比

## 执行顺序

1. US-001 (基准测量) - 无依赖
2. US-002 (KV Cache) - 依赖 US-001
3. US-003 (同步点) - 依赖 US-001
4. US-004 (Token 分配) - 依赖 US-001
5. US-005 (RMSNorm) - 依赖 US-001
6. US-006 (CUDA Graph) - 依赖 US-001
7. US-007 (验证) - 依赖 US-002, US-003, US-004, US-005, US-006
