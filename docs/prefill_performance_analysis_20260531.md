# Rust Server Prefill 性能分析与优化方案

## 测试模型
`/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ`

## Python TurboMind Prefill 基准数据（真实单请求）

实测数据（2026-05-31）：

| Input | TTFT avg(ms) | TTFT std | Prefill avg (tok/s) | Prefill max (tok/s) |
|-------|--------------|---------|---------------------|---------------------|
| 512 | 116.8 | 6.3 | 4,398 | 5,230 |
| 1024 | 189.8 | 6.9 | 5,404 | 6,036 |
| 2048 | 334.6 | 5.7 | 6,122 | 6,450 |
| 4096 | 674.0 | 5.7 | 6,078 | 6,232 |
| 8192 | 1560.8 | 7.0 | 5,249 | 5,318 |

**说明**: 
- 这些是真实的单请求 prefill 性能，不是并发聚合吞吐量
- 用户看到的"好几万 tok/s"可能是并发场景下的 aggregate 吞吐量
- 测试方法：10次运行取平均，测量从请求开始到第一个token的时间（TTFT）
- Prefill tok/s = input_len / TTFT

## Rust Server 当前状态

### 已完成的优化
1. ✅ **异步 forward 模式**: `generate_with_gpu_tokenizer_and_metrics()` 已从阻塞 `request.forward()` 改为 `forward_async()` + completion callback
2. ✅ **GPU tokenizer 零拷贝路径**: 已实现 GPU tokenizer 直接 tokenize 到 pinned memory
3. ✅ **uint32 零拷贝传输**: `set_input_ids_gpu_uint32_async()` 避免了 u32->i64 转换

### 关键阻塞问题
**❌ C++ InitFromPath 崩溃**: 加载 35B AWQ 模型时，CreateEngine 阶段崩溃
- 错误: `'data_' Must be non NULL` at buffer.h:70
- 位置: ProcessWeights 或 CreateEngine 中的 buffer 访问
- 影响: 无法运行 Rust server 进行实际性能对比测试

## 根本原因分析

### C++ 加载路径差异
**Python TurboMind**:
- Python 直接调用 C++ 内部 API
- 权重通过 Python CFFI/ctypes 加载后传递给 C++
- C++ ModelWeight 直接使用已分配的张量

**Rust C API**:
- 通过 C API `TM_TurboMind_InitFromPath` 加载
- 使用 `LoadWeightsFromSafetensors` 直接从 safetensors 读取
- 通过 `ProcessWeights` + `CreateEngine` 初始化引擎

### 可能的崩溃点
1. **ProcessWeights.prepare()**: 访问未分配的张量（如 output->output_dim）
2. **Context/Allocator 不匹配**: managed allocator guard 作用域问题
3. **权重张量引用**: safetensors 加载后张量生命周期问题

## 优化方案

### Phase 1: 修复 C++ 加载崩溃（P0 - 阻塞）

**目标**: 让 Rust server 能成功加载 35B AWQ 模型

**调查方向**:
1. 在 `ProcessWeights.prepare()` 添加调试，检查 `output` 模块是否有效
2. 验证 `LoadWeightsFromSafetensors` 后所有张量是否正确分配
3. 检查 `ContextGuard` 生命周期和 allocator 切换
4. 对比 Python 加载路径，找出差异

**预期修复**: 修改 `turbomind_c.cc` 或 `model_weight.cc` 中的张量访问逻辑

### Phase 2: 性能验证与对比

**目标**: 测量 Rust server prefill 性能，与 Python 对比

**测试方法**:
1. 启动 Rust server（需先修复 C++ 加载）
2. 运行 `benchmark/comprehensive_prefill_bench.py` 的 Rust 版本
3. 对比相同输入长度下的 TTFT 和 prefill tok/s

**成功标准**: Rust prefill ≥ Python prefill (10% 误差内)

### Phase 3: 进一步优化（如需要）

**如果 Rust 仍然慢**:
1. **分析热点**: 使用 `nvprof`/`nsys` 定位瓶颈
2. **优化 tokenizer**: 确保 GPU tokenizer 路径被使用
3. **减少同步点**: 检查是否有不必要的 CUDA event sync
4. **内存池优化**: 预分配 buffer，减少 per-request 分配

## Beads 任务

### 当前任务
- [lmdeploy-lin8] P0 bug: 分析并修复 ProcessWeights/CreateEngine 崩溃 - data_ NULL buffer
  - 状态: open
  - 依赖: lmdeploy-p9lz

### 已完成任务
- [lmdeploy-wh5q] P1 task: 分析 Rust Server vs Python TurboMind prefill 性能差距 ✅
- [lmdeploy-8w7q] P1 task: Rust 实现异步 forward_async 替代阻塞 forward ✅

### 待创建任务
- [ ] 修复 C++ 张量访问逻辑 - 确保 prepare() 前所有张量已分配
- [ ] Rust prefill 性能基准测试 - 与 Python 对比
- [ ] 性能热点分析 - 如需要，使用 nvprof 定位瓶颈

## 下一步行动

1. **立即**: 修复 C++ 加载崩溃（阻塞问题）
2. **然后**: 运行 Rust prefill 基准测试
3. **最后**: 根据测试结果决定是否需要进一步优化

## 数据文件

- Python 基准: `python_prefill_benchmark_20260531_023222.json`
- 测试脚本: `benchmark/comprehensive_prefill_bench.py`
- Rust 代码: `lmdeploy-rust-server/src/model/cpp_engine.rs` (line 1846-1987)
