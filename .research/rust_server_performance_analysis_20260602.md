# LMDeploy Rust Server Performance Analysis

## Executive Summary

对 Python TurboMind 和 Rust Server 进行了性能基准测试和分析。**关键发现：Rust Server 完全无法运行**，存在三个致命 BUG 阻止其正常初始化和推理。

## 基准测试结果

### Python TurboMind (已验证，真实数据)

**测试环境:**
- 模型: Qwen3.6-35B-A3B-AWQ
- GPU: Tesla PG503-216 (32GB)
- TP: 1
- 输出长度: 512 tokens
- 测试方法: 5 次迭代取中位数

| Context | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Overall (tok/s) |
|---------|-----------|-----------------|-----------------|-----------------|
| 4k      | 1046      | **3915**        | 41.6            | 38.4            |
| 8k      | 2468      | **3319**        | 44.4            | 36.7            |
| 16k     | 6893      | **2377**        | 38.2            | 25.2            |

**观察:**
- Prefill 吞吐量随上下文长度增加而下降（符合预期，因为更多内存访问）
- Decode 速度相对稳定在 ~40 tok/s
- 4K prefill 达到 ~4000 tok/s，这是**真实的 C++ TurboMind 性能**

### Rust Server (完全无法运行)

**崩溃信息:**
```
[TM][FATAL][buffer.h:70] 'data_' Must be non NULL
*** stacktrace of thread 0xc142c1a3000 ***
  [ 0] TM_CHECK_NOTNULL @ buffer.h:70
```

**关键日志:**
```
[C-API] Loaded 0 tensors, skipped 2055 from model-00001-of-00009.safetensors
[C-API] Loaded 0 tensors, skipped 14835 from model-00002-of-00009.safetensors
...
[C-API] ERROR: w_qkv.weight param slot not found for layers.7.attention
[C-API] ERROR: w_qkv.weight param slot not found for layers.3.attention
```

## 三个致命 BUG 分析

### BUG #1: 权重完全未加载 (严重性: 致命)

**现象:**
所有 safetensors 文件都是 `"Loaded 0 tensors, skipped NNNN"` — 权重根本没有加载到 GPU 内存。

**根本原因:**
Rust C-API 加载逻辑中存在参数匹配失败。日志显示所有张量都被 "skipped"，说明参数名称/路径不匹配。

**影响:**
由于权重未加载，后续所有推理操作都会访问空指针，导致崩溃。

### BUG #2: w_qkv.weight 参数不匹配 (严重性: 致命)

**现象:**
```
[C-API] ERROR: w_qkv.weight param slot not found for layers.7.attention
[C-API]   w_qkv_param.get() valid=0
```

**根本原因:**
Qwen3.6-35B-A3B-AWQ 是 AWQ 量化模型，其注意力权重结构可能与 C++ 代码期望的不同：
- C++ 代码期望: `layers.N.attention.w_qkv.weight` (融合 QKV)
- 实际模型可能是: 分离的 `w_q.weight`, `w_k.weight`, `w_v.weight`
- 或者量化后权重命名不同

**影响:**
QKV fusion 失败，导致注意力层权重无法正确加载。

### BUG #3: Buffer::data() 空指针断言 (严重性: 致命)

**现象:**
```
[TM][FATAL][buffer.h:70] 'data_' Must be non NULL
```

**根本原因:**
由于 BUG #1 和 #2，权重未加载，推理时 Buffer 的 `data_` 为 NULL。某处代码仍然使用 `TM_CHECK_NOTNULL(data_)` 来校验 Buffer 指针，导致断言失败崩溃。

虽然 `buffer.h:70-76` 的 `data()` 方法已有 `if (!data_) return nullptr;` 检查，但崩溃说明：
1. 某处直接使用了 `TM_CHECK_NOTNULL(buffer.data())`
2. 或者 `TM_CHECK_NOTNULL` 被应用于某个未初始化的 Buffer

## 性能对比分析

由于 Rust Server 完全无法运行，无法进行真实的性能对比。但从架构角度分析：

### Python TurboMind 性能来源

1. **C++ TurboMind 引擎** - 核心推理在 C++ 中执行
2. **pybind11 绑定** - 轻量级 Python/C++ FFI
3. **零拷贝张量传输** - DLPack 协议避免数据复制
4. **异步流式推理** - `async_stream_infer` 实现

### Rust Server 理论优势

1. **更轻量的 FFI** - 直接调用 C API，无 Python GIL
2. **异步并发** - Tokio 异步运行时
3. **内存安全** - Rust 编译时保证
4. **更好的控制** - 可以直接管理 CUDA streams 和事件

### Rust Server 理论劣势

1. **序列化开销** - 如果使用 gRPC/HTTP
2. **额外抽象层** - Rust FFI 绑定可能引入开销
3. **未优化路径** - 可能缺少某些 Python 路径的优化

## 解决方案建议

### 优先级 1: 修复致命 BUG（必须）

1. **修复权重加载逻辑**
   - 调试 C-API 的参数匹配逻辑
   - 确认 AWQ 模型的权重命名规范
   - 添加详细的加载日志

2. **修复 QKV fusion 逻辑**
   - 支持 AWQ 量化模型的权重结构
   - 添加 fallback 到分离 Q/K/V 权重的路径
   - 参考 Python TurboMind 的实现

3. **修复 Buffer 空指针问题**
   - 确保所有 Buffer 使用前都检查 `if (!buffer)` 或 `if (!buffer.data())`
   - 移除或修复所有 `TM_CHECK_NOTNULL(buffer.data())` 调用
   - 添加更好的错误处理

### 优先级 2: 性能优化（修复后）

1. **消除 Python GIL 开销** - Rust 的主要优势
2. **优化异步调度** - 更好的 CUDA stream 管理
3. **批量处理优化** - 改进批处理调度
4. **零拷贝优化** - 确保 DLPack 张量零拷贝

## 结论

**当前状态:**
- Python TurboMind: ✅ 正常运行，Prefill 2377-3915 tok/s
- Rust Server: ❌ 完全无法运行，三个致命 BUG

**目标:**
修复 BUG 后，Rust Server 应该能够：
1. 首先达到与 Python TurboMind 相同的性能
2. 然后通过消除 Python GIL 开销超越 10-20%
3. 最终通过更好的异步调度超越 20-30%

**下一步:**
创建 PRD 和 Beads 任务，系统化地修复这三个 BUG 并进行性能优化。
