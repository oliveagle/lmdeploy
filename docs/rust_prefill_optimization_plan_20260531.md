# Rust Server Prefill 性能优化任务拆解

## 日期: 2026-05-31

## Python TurboMind 真实性能数据（2026-05-31 实测）

两次独立运行，数据极其稳定：

| Input | TTFT (ms) | Prefill (tok/s) |
|-------|-----------|-----------------|
| 512 | 117.8 ± 0.7 | 4,362 |
| 1024 | 188.2 ± 1.0 | 5,447 |
| 2048 | 336.8 ± 1.4 | 6,082 |
| 4096 | 676.9 ± 1.5 | 6,052 |
| 8192 | 1560.8 ± 1.4 | 5,249 |

**关键发现**: Python 实测 prefill 最高 ~6,000 tok/s，不是"好几万"。用户看到的高数值可能是：
1. 并发 aggregate 吞吐量（多请求总和）
2. 不同硬件配置
3. 早期基准测试使用不同模型/参数

**实际目标**: Rust prefill 性能需与上述数据匹配（10% 误差内），而非"好几万"。

---

## 当前状态分析

### P0 阻塞问题: C++ 加载崩溃

**症状**: Rust server 调用 `TM_TurboMind_InitFromPath` 时崩溃
**错误**: `'data_' Must be non NULL` at buffer.h:76
**位置**: `ModelWeight::prepare()` 中访问已分配权重的 buffer data

**根本原因分析**:
1. `LoadWeightsFromSafetensors()` 使用 `cudaMallocManaged` 分配权重
2. `batch_copy.Run()` 将数据从 mmap 拷贝到 GPU
3. 但 `tm_instance.CreateContext(0)` 调用 `CreateContext(index)` 后
4. `ProcessWeights(index)` 调用 `weights_[index]->prepare()` 时，某个模块的 buffer data_ 为 null

**关键代码路径差异**:
- **Python**: 直接 pybind11 调用 → C++ 内部 API → 权重通过 C++ 内部 `LoadWeights` 加载
- **Rust**: C API `TM_TurboMind_InitFromPath` → `LoadWeightsFromSafetensors` → `ProcessWeights` → 崩溃

**可能原因**:
1. **上下文切换丢失**: `LoadWeightsFromSafetensors` 使用 `managed_allocator` 创建 tensor，但后续调用中上下文 allocator 被切换回默认的 CUDA allocator，导致 buffer data_ 不可访问
2. **张量未完全分配**: safetensors 加载时某些张量的 `target_param` slot 不存在（`!target_param.get()` 检查跳过）
3. **batch_copy 未完成**: 拷贝操作是异步的，可能在拷贝完成前就调用了 prepare()

---

## Beads 任务规划

### Phase 0: P0 Bug 修复（阻塞一切）

#### Beads 1: 修复 C++ 权重加载崩溃（P0）
**描述**: Rust server 无法加载 35B AWQ 模型，InitFromPath 阶段崩溃
**依赖**: 无
**验收标准**: Rust server 能成功加载 35B AWQ 模型并启动

**实施步骤**:
1. 在 `turbomind_c.cc` 的 `LoadWeightsFromSafetensors` 中添加详细调试日志，记录每个张量的分配和拷贝状态
2. 验证 `batch_copy.Run()` 是同步完成还是异步（如果是异步，需要添加同步点）
3. 检查 `ContextGuard` 和 allocator 切换逻辑是否正确
4. 对比 Python 加载路径，确保 Rust 的加载路径完整覆盖所有必需的张量
5. 修复找到的任何张量分配或上下文问题

**涉及文件**:
- `src/turbomind/capi/turbomind_c.cc`
- `src/turbomind/models/model_weight.cc`

#### Beads 2: 添加详细加载调试工具（P1）
**描述**: 为 safetensors 加载添加详细的调试输出，方便定位问题
**依赖**: Beads 1 完成后删除
**验收标准**: 能输出每个张量的加载、分配、拷贝状态

---

### Phase 1: 性能分析与基准确认

#### Beads 3: Rust vs Python 性能基准对比（P1）
**描述**: 在修复 C++ 加载崩溃后，运行 Rust 和 Python 的 prefill 基准对比测试
**依赖**: Beads 1
**验收标准**: 得到 Rust 和 Python 的 TTFT、prefill tok/s 对比数据

**实施步骤**:
1. 修复 C++ 加载崩溃后，确保 Rust server 能运行
2. 创建统一基准测试脚本，测试相同的输入长度
3. 测量 TTFT、prefill throughput
4. 对比 Python 和 Rust 的结果，确认性能差距
5. 如果 Rust 已经接近 Python（10% 误差），无需进一步优化

#### Beads 4: 热点分析（P1）
**描述**: 如果 Rust 比 Python 慢 >10%，使用性能分析工具定位瓶颈
**依赖**: Beads 3
**验收标准**: 输出详细的热点分析报告

**分析工具**:
- `nsys` / `nvprof`: GPU kernel 级别
- `perf`: CPU 级别
- 内部计时: Rust 和 C++ 关键路径

**分析范围**:
- Tokenization 耗时
- GPU 内存拷贝耗时
- C++ forward 耗时
- 同步点开销

---

### Phase 2: 快速优化（基于热点分析结果）

#### Beads 5: 优化 GPU Tokenizer 路径（P2）
**描述**: 确保 GPU tokenizer 路径被使用，消除 CPU tokenizer 的额外拷贝
**依赖**: Beads 3, Beads 4
**验收标准**: Tokenizer 路径耗时与 Python 相当

#### Beads 6: 消除不必要的 Event Sync（P2）
**描述**: 移除重复的 `event.sync()` 调用
**依赖**: Beads 3, Beads 4
**验收标准**: 减少 5-10ms per request

#### Beads 7: 优化 Condvar Timeout（P2）
**描述**: 将 `wait_timeout` 改为 `wait`，消除不必要的超时延迟
**依赖**: Beads 3, Beads 4
**验收标准**: 减少 0-10ms per request

---

### Phase 3: 高级优化（如需要）

#### Beads 8: 异步 Forward 路径（P2）
**描述**: 对非流式路径使用 `forward_async` 替代阻塞 `forward`
**依赖**: Beads 4
**验收标准**: 减少 1.5-2x forward 耗时

#### Beads 9: Request Pool 优化（P3）
**描述**: 在 tokenization 期间预获取 pool slot，重叠 CPU/GPU 工作
**依赖**: Beads 4
**验收标准**: 减少 1-2ms per request

#### Beads 10: 内存池优化（P3）
**描述**: 增加 GPU tokenizer buffer size，优化大 prompt 处理
**依赖**: Beads 4
**验收标准**: >8K token contexts 性能提升

---

## 成功标准

1. **P0**: Rust server 能成功加载 35B AWQ 模型并启动（阻塞一切）
2. **P1**: Rust prefill 性能与 Python 匹配（10% 误差内）
3. **P2**: 无内存泄漏，压力下无死锁（1000+ 请求）
4. **P3**: 现有 API 不变

## 预计时间线

- **Phase 0** (1-2 天): 修复 C++ 加载崩溃
- **Phase 1** (1 天): 性能基准确认
- **Phase 2** (2-3 天): 快速优化（如有需要）
- **Phase 3** (1-2 周): 高级优化（如需要）
