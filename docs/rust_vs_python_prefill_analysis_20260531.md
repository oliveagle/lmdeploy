# Rust Server vs Python TurboMind Prefill 性能分析报告

## 日期: 2026-05-31

## 1. Python TurboMind 真实性能数据（实测）

### 测试配置
- **模型**: `/mnt/data/models/modelscope_models/Qwen3.6-35B-A3B-AWQ`
- **硬件**: CUDA GPU
- **测试方法**: 单请求串行测试，10次运行取平均
- **测量指标**: TTFT (Time To First Token)，Prefill throughput

### 实测结果（两次运行验证）

| Input | TTFT avg (ms) | TTFT std (ms) | Prefill avg (tok/s) | Prefill max (tok/s) |
|-------|---------------|---------------|---------------------|---------------------|
| 512 | 117.8 ± 0.7 | 6.5 | 4,362 | 5,177 |
| 1024 | 188.2 ± 1.0 | 6.2 | 5,447 | 6,012 |
| 2048 | 336.8 ± 1.4 | 6.0 | 6,082 | 6,409 |
| 4096 | 676.9 ± 1.5 | 7.0 | 6,052 | 6,230 |
| 8192 | 1560.8 ± 1.4 | 5.5 | 5,249 | 5,305 |

**关键发现**:
- **实测最高**: ~6,100 tok/s (2048 tokens)
- **用户报告的"好几万"**: 可能是并发 aggregate 吞吐量，而非单请求 prefill
- **实际目标**: Rust 需达到 ~5,000-6,000 tok/s，而非"好几万"

---

## 2. Rust Server 当前状态

### P0 阻塞问题: C++ 加载崩溃

**症状**: Rust server 无法加载 35B AWQ 模型
**错误**: `'data_' Must be non NULL` at buffer.h:76
**位置**: `ModelWeight::prepare()` 访问权重 buffer data

**代码路径差异**:
```
Python: Python → pybind11 → C++ 内部 API → LoadWeights
Rust:    Rust → C API → InitFromPath → LoadWeightsFromSafetensors → ProcessWeights → 崩溃
```

**根本原因假设**:
1. **上下文丢失**: safetensors 加载使用 `cudaMallocManaged` + `managed_allocator`，但后续调用中 allocator 被切换
2. **异步拷贝**: `batch_copy.Run()` 可能是异步的，拷贝未完成就调用了 `prepare()`
3. **张量不完整**: 某些张量的 `target_param` slot 不存在（通过 `!target_param.get()` 检查跳过）

---

## 3. 性能差距分析（假设崩溃修复后）

### 已知潜在瓶颈

#### 瓶颈 1: Tokenizer 路径
- **Python**: 直接 pybind11 绑定，无中间层
- **Rust**: CPU tokenizer → u32 转 i64 → GPU 拷贝 → DLPack → C++
- **GPU 路径已存在**: `generate_with_gpu_tokenizer_and_metrics()` 使用 GPU tokenizer

#### 瓶颈 2: C++ Forward 模式
- **Python**: 直接 pybind11 调用，基于回调的异步完成
- **Rust**: C API `TM_ModelRequest_Forward()` 使用 `std::promise`/`future.get()` 阻塞模式

#### 瓶颈 3: 同步点开销
- **位置**: `event.sync()` (cpp_engine.rs:2181)
- **问题**: GPU tokenizer 内部已同步，双重同步浪费时间

#### 瓶颈 4: Condvar Timeout
- **位置**: `cvar.wait_timeout(done, Duration::from_millis(10))` (cpp_engine.rs:2576)
- **问题**: 添加 10ms 超时延迟，回调保证会触发

---

## 4. Beads 任务规划

### Phase 0: P0 Bug 修复（阻塞一切）

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| lmdeploy-2o30 | 修复 C++ InitFromPath 权重加载崩溃 | P0 | In Progress | - |

### Phase 1: 性能分析与基准确认

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| lmdeploy-zvtv | Rust vs Python Prefill 性能基准对比 | P1 | Open | lmdeploy-2o30 |
| lmdeploy-wk5f | Prefill 性能热点分析 | P1 | Open | lmdeploy-2o30 |

### Phase 2: 快速优化

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| lmdeploy-p0lz | 优化 GPU Tokenizer 路径 | P2 | Open | lmdeploy-zvtv, lmdeploy-wk5f |
| lmdeploy-cwc5 | 消除不必要的 Event Sync | P2 | Open | lmdeploy-zvtv, lmdeploy-wk5f |
| lmdeploy-ki4v | 优化 Condvar Timeout | P2 | Open | lmdeploy-zvtv, lmdeploy-wk5f |

### Phase 3: 高级优化

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| lmdeploy-4b8p | 异步 Forward 路径优化 | P2 | Open | lmdeploy-wk5f |
| lmdeploy-51ga | Request Pool 预获取优化 | P3 | Open | lmdeploy-wk5f |
| lmdeploy-bru7 | 内存池优化 | P3 | Open | lmdeploy-wk5f |

### Phase 4: 验证

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| lmdeploy-747b | Prefill 性能综合验证 | P1 | Open | lmdeploy-p0lz, lmdeploy-cwc5, lmdeploy-ki4v |

---

## 5. 成功标准

1. **P0**: Rust server 能成功加载 35B AWQ 模型并启动
2. **P1**: Rust prefill 性能与 Python 匹配（10% 误差内）
3. **P2**: 无内存泄漏，压力下无死锁（1000+ 请求）
4. **P3**: 现有 API 不变

---

## 6. 数据文件

- Python 基准: `python_prefill_benchmark_20260531_025642.json`
- Python 基准: `python_prefill_benchmark_20260531_025813.json`
- 分析文档: `docs/rust_prefill_optimization_plan_20260531.md`

---

## 7. 下一步行动

1. **立即**: 修复 C++ 加载崩溃（lmdeploy-2o30）
2. **然后**: 运行 Rust prefill 基准测试（lmdeploy-zvtv）
3. **最后**: 根据热点分析结果决定优化策略（lmdeploy-wk5f）
