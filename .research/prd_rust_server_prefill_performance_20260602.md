# PRD: Rust Server 超越 Python TurboMind Prefill 性能

## 概述

**目标:** 修复 Rust Server 的三个致命 BUG，使其能够正常运行，然后优化 prefill 性能超越 Python TurboMind 20%以上。

**当前状态:**
- Python TurboMind: ✅ 正常运行，Prefill 2377-3915 tok/s (已验证)
- Rust Server: ❌ 完全无法运行，崩溃在 Buffer 空指针

**目标性能:**
| Context | Python (tok/s) | Rust 目标 (tok/s) | 提升比例 |
|---------|----------------|-------------------|----------|
| 4k      | 3915           | 4700+             | +20%     |
| 8k      | 3319           | 4000+             | +20%     |
| 16k     | 2377           | 2850+             | +20%     |

## 用户故事

### Story 1: 修复权重加载 BUG

**描述:** 作为开发者，我需要 Rust Server 能够正确加载 AWQ 量化模型的权重，以便进行推理。

**验收标准:**
- [ ] 所有 safetensors 文件的张量能够正确加载（不再全部 "skipped"）
- [ ] 权重加载日志显示 "Loaded NNNN tensors" 而非 "Loaded 0 tensors"
- [ ] 引擎初始化成功，无崩溃

**技术细节:**
- 调试 `src/turbomind/capi/turbomind_c.cc` 中的权重加载逻辑
- 确认参数名称匹配规则
- 添加详细日志追踪参数查找过程

### Story 2: 修复 QKV Fusion BUG

**描述:** 作为开发者，我需要 Rust Server 能够正确处理 AWQ 模型的注意力权重结构，以便加载 QKV 权重。

**验收标准:**
- [ ] 不再出现 "w_qkv.weight param slot not found" 错误
- [ ] QKV fusion 成功或正确 fallback 到分离 Q/K/V
- [ ] 所有 40 层的注意力权重正确加载

**技术细节:**
- 修复 `turbomind_c.cc:1222-1227` 的 QKV fusion 逻辑
- 支持分离的 Q/K/V 权重格式
- 参考 Python TurboMind 的实现

### Story 3: 修复 Buffer 空指针 BUG

**描述:** 作为开发者，我需要 Rust Server 能够安全处理未初始化的 Buffer，避免空指针崩溃。

**验收标准:**
- [ ] 不再出现 `[TM][FATAL][buffer.h:70] 'data_' Must be non NULL` 崩溃
- [ ] 所有 Buffer 使用前都有 `if (!buffer)` 或 `if (!buffer.data())` 检查
- [ ] 推理能够正常完成，返回结果

**技术细节:**
- 审计所有 `TM_CHECK_NOTNULL(buffer.data())` 调用
- 添加安全的空指针检查
- 改进错误处理和日志

### Story 4: 验证基准性能

**描述:** 作为开发者，我需要验证 Rust Server 修复后能够达到与 Python TurboMind 相当的性能。

**验收标准:**
- [ ] Rust Server 能够成功运行 4k/8k/16k prefill 基准测试
- [ ] 性能达到 Python TurboMind 的 90-100%
- [ ] 基准结果可重复且稳定

**技术细节:**
- 运行 `prefill_benchmark_4_8_16k`
- 对比 Python 和 Rust 的 TTFT、Prefill TPS、Decode TPS
- 分析性能差异原因

### Story 5: 优化 Prefill 性能超越 Python

**描述:** 作为开发者，我需要优化 Rust Server 的 prefill 性能，使其超越 Python TurboMind 20%以上。

**验收标准:**
- [ ] 4K prefill 超过 4700 tok/s (+20%)
- [ ] 8K prefill 超过 4000 tok/s (+20%)
- [ ] 16K prefill 超过 2850 tok/s (+20%)
- [ ] Decode 性能不低于 Python

**优化方向:**
1. **消除 Python GIL 开销** - Rust 的主要优势
2. **优化异步调度** - 更好的 CUDA stream 管理
3. **减少序列化开销** - 零拷贝优化
4. **批量处理优化** - 改进批处理调度
5. **GPU Tokenizer 优化** - 确保 GPU tokenizer 零拷贝

## 技术架构

### 当前架构问题

```
Rust Server → C API → TurboMind C++
              ↓ FFI
         权重加载失败 (BUG #1)
         QKV fusion 失败 (BUG #2)
         Buffer 空指针 (BUG #3)
```

### 目标架构

```
Rust Server → C API → TurboMind C++ (与 Python 相同)
              ↓ FFI
         正确加载权重
         正确处理 QKV
         安全处理 Buffer
              ↓
         性能优化层 (超越 Python)
```

## 实现计划

### Phase 1: BUG 修复 (必须)

1. **调试权重加载** (1-2 天)
   - 添加详细日志
   - 追踪参数查找
   - 找到 "skipped" 原因

2. **修复 QKV fusion** (1-2 天)
   - 支持 AWQ 权重格式
   - 添加 fallback 路径
   - 测试所有层

3. **修复 Buffer 空指针** (1 天)
   - 审计所有 Buffer 使用
   - 添加安全检查
   - 测试推理流程

### Phase 2: 性能验证 (1 天)

1. 运行基准测试
2. 对比 Python 性能
3. 分析性能差距

### Phase 3: 性能优化 (2-3 天)

1. 消除序列化开销
2. 优化异步调度
3. GPU tokenizer 零拷贝
4. 批量处理优化

## 测试计划

### 单元测试

- [ ] 权重加载测试
- [ ] QKV fusion 测试
- [ ] Buffer 安全测试

### 集成测试

- [ ] 端到端推理测试
- [ ] 4k/8k/16k prefill 测试
- [ ] 批量推理测试

### 性能测试

- [ ] 与 Python TurboMind 对比
- [ ] 多次运行取平均值
- [ ] 不同 batch size 测试

## 风险与依赖

### 风险

1. **AWQ 权重格式复杂** - 可能需要深入理解量化格式
2. **C++ 代码复杂** - TurboMind C++ 代码量大，调试困难
3. **性能优化困难** - 可能需要 CUDA 级别优化

### 依赖

1. **C++ TurboMind 正常工作** - Python 版本已验证
2. **AWQ 模型正确** - 模型文件完整
3. **GPU 资源** - 32GB 显存足够

## 验收标准

**总体验收:**
- [ ] Rust Server 能够正常运行 4k/8k/16k prefill 基准测试
- [ ] Prefill 性能超越 Python TurboMind 20%以上
- [ ] Decode 性能不低于 Python
- [ ] 无崩溃，无内存泄漏
- [ ] 代码通过 review

**性能验收:**
| Context | Python 目标 | Rust 目标 | 实际 | 状态 |
|---------|--------------|-----------|------|------|
| 4k      | 3915 tok/s   | 4700+     | TBD  | TBD  |
| 8k      | 3319 tok/s   | 4000+     | TBD  | TBD  |
| 16k     | 2377 tok/s   | 2850+     | TBD  | TBD  |

## 交付物

1. 修复的 Rust Server 代码
2. 基准测试结果报告
3. 性能优化文档
4. 单元测试和集成测试

## 参考资料

- 性能分析: `.research/rust_server_performance_analysis_20260602.md`
- Python 基准: `benchmark/benchmark_pipeline_4k_8k_16k.py`
- Rust 基准: `lmdeploy-rust-server/src/bin/prefill_benchmark_4_8_16k.rs`
- C++ C API: `src/turbomind/capi/turbomind_c.cc`
- Buffer 实现: `src/turbomind/core/buffer.h`
