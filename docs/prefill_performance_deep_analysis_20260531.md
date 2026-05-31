# Prefill 性能深度分析报告

## 日期: 2026-05-31

---

## 1. 核心问题

用户观察到 prefill 吞吐量"好几万 tok/s"，但实测只有 4,000-6,200 tok/s。

---

## 2. 真实性能数据（2026-05-31 实测，5 次运行平均）

### Python + TurboMind C++

| Input | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|-------|-----------|-----------------|----------------|
| 512 | 118.7 | 4,312 | 41.7 |
| 1024 | 189.8 | 5,394 | 41.2 |
| 4096 | 676.3 | 6,057 | 38.5 |
| 8192 | 1564.6 | 5,236 | 35.4 |

### Archive 声称（2026-05-26）

| Input | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|-------|-----------|-----------------|----------------|
| 512 | 80.0 | 6,408 | 42.6 |
| 1024 | 139.2 | 7,356 | 42.2 |
| 4096 | 402.0 | 10,190 | 40.5 |
| 8192 | 649.4 | 12,614 | 39.8 |

### 性能差距

| Input | 当前 vs Archive | 差距 |
|-------|-----------------|------|
| 512 | 4,312 vs 6,408 | **-48%** |
| 1024 | 5,394 vs 7,356 | **-36%** |
| 4096 | 6,057 vs 10,190 | **-68%** |
| 8192 | 5,236 vs 12,614 | **-141%** |

### 验证 Archive 提交 (738f60eb) 实测

回到 Archive 对应的 git commit 738f60eb 重新测试，结果为:
- 512: 898 tok/s (TTFT 1140ms)
- 1024: 1076 tok/s (TTFT 3805ms)
- 4096: 868 tok/s (TTFT 9443ms)
- 8192: 841 tok/s (TTFT 9741ms)

**关键发现**: Archive 提交点的 README 声称的 6,408-12,614 tok/s 与同一提交点实测的 841-1076 tok/s **完全不符**！

**结论**: Archive 的 README 中声称的高性能数据**不是来自 Python + TurboMind 单请求测试**，很可能是来自其他测试工具（如 benchmark.py 的 profile_throughput 模式）的 aggregate 吞吐量，或者是使用了完全不同的测试方法。

### 真正的性能目标

| Input | 真实性能目标 |
|-------|-------------|
| 512 | ~4,000-5,000 tok/s |
| 1024 | ~5,000-6,000 tok/s |
| 4096 | ~5,000-6,500 tok/s |
| 8192 | ~5,000-5,500 tok/s |

**Rust 必须超过的目标**: 峰值 ~6,000 tok/s，不是"好几万"。

---

## 3. "好几万 tok/s" 的可能来源

1. **Aggregate 吞吐量**: `benchmark/profile_throughput.py` 报告的 `input_throughput` 是 aggregate 指标（多个请求并发时的总吞吐量），不是单请求 prefill
2. **Decode 阶段**: 单请求 Decode 约 40 tok/s，但如果 100 个并发请求，aggregate 可达 ~4000 tok/s
3. **测试工具差异**: 不同测试工具可能有不同测量点（从 token 回调 vs 从 forward 结束）

---

## 4. Rust Server 性能分析

### 当前状态
- **lmdeploy-2o30**: C++ 崩溃已修复 ✅
- **lmdeploy-zvtv**: Python 基准已采集 ✅
- **lmdeploy-p0lz**: GPU Tokenizer 优化 🔄 in_progress
- **lmdeploy-ki4v**: Condvar 优化 ✅ (已关闭)
- **lmdeploy-cwc5**: Event Sync 消除 ✅ (已关闭)

### 已实施优化

1. **forward_async**: 异步 forward 替代阻塞 promise/future
2. **GPU Tokenizer**: 消除 CPU tokenizer 到 GPU 的拷贝
3. **uint32 零拷贝**: input_ids 直接从 tokenizer uint32 拷贝到 GPU，跳过 i64 转换
4. **Event-driven wait**: Condvar wait() 替代 timeout 轮询
5. **Request Pool**: 预分配的 ModelRequest 对象池

### 剩余瓶颈分析

#### 瓶颈 A: GPU 内存拷贝序列化
```rust
// cpp_engine.rs:2180
if let Some(event) = set_input_ids_gpu_uint32_async(...) {
    let _ = event.sync();  // ← 阻塞等待 GPU 拷贝完成
}
// 然后才调用 forward_async
```

即使使用 `forward_async`，在 GPU 数据拷贝完成前必须 sync，这引入了额外的同步点。

#### 瓶颈 B: Condvar 锁竞争
```rust
// 每个请求都创建一个 Condvar 并 wait
// 在高并发下，多个请求的 Condvar 等待可能造成锁竞争
```

#### 瓶颈 C: TensorMap 重复创建
```rust
// 每次请求都创建新的 TensorMap + cudaMemcpy
// 虽然用了 TensorMap pool，但仍有额外的拷贝开销
```

#### 瓶颈 D: 回调链路
```
C++ Gateway → Token callback → Rust token_cb_wrapper → Rust state → Condvar notify
```

Python 的 pybind11 路径可能更直接。

---

## 5. 超越 Python 的具体策略

### 策略 1: 消除所有 Event Sync（预期提升 5-10%）

当前: `event.sync()` → `forward_async()` → `Condvar.wait()`
目标: `streamed_copy_async()` → `forward_async()` → GPU 内部自动同步

### 策略 2: C++ Gateway 层直接接入（预期提升 10-20%）

当前: Rust → C API → ModelRequest::Forward → Gateway → Engine
目标: Rust → C API → Gateway 直接提交 → 跳过 ModelRequest 层

### 策略 3: CUDA Stream 重叠（预期提升 5-10%）

使用不同的 CUDA stream 进行 H2D 拷贝和计算，让拷贝和计算重叠。

### 策略 4: Batch 优化（预期提升 20-50%）

Python 使用 Gateway 的 batch 调度，Rust 也需实现类似的 batch 提交机制。

---

## 6. Beads 任务状态

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| lmdeploy-2o30 | 修复 C++ 崩溃 | P0 | ✅ Closed | - |
| lmdeploy-zvtv | Python 基准对比 | P1 | ✅ Closed | lmdeploy-2o30 |
| lmdeploy-ki4v | Condvar 优化 | P2 | ✅ Closed | lmdeploy-zvtv |
| lmdeploy-cwc5 | Event Sync 消除 | P2 | ✅ Closed | lmdeploy-zvtv |
| lmdeploy-p0lz | GPU Tokenizer 优化 | P2 | 🔄 In Progress | lmdeploy-zvtv |
| lmdeploy-747b | Prefill 综合验证 | P1 | Open | lmdeploy-p0lz, lmdeploy-cwc5 |

### 待创建任务

| ID | 任务 | 优先级 | 依赖 |
|----|------|--------|------|
| NEW | "好几万 tok/s" 根因分析 | P1 | - |
| NEW | Rust 超过 Python prefill 优化方案 | P1 | lmdeploy-p0lz |
| NEW | C++ Gateway 直接接入优化 | P2 | lmdeploy-747b |
| NEW | CUDA Stream 重叠优化 | P3 | lmdeploy-747b |

---

## 7. 下一步

1. 等待 lmdeploy-p0lz 完成（GPU Tokenizer 优化）
2. 创建"好几万 tok/s" 根因分析 beads
3. 创建 Rust 超过 Python prefill 优化 beads
4. 在 lmdeploy-p0lz 完成后运行 Rust 基准测试
