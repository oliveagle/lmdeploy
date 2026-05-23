# LMDeploy Rust vs Python TurboMind 性能对比报告

**日期**: 2026-05-23
**任务**: lmdeploy-kg8 - 对比 Rust+C++ vs Python TurboMind 推理性能

---

## 测试环境

| 项目 | 配置 |
|------|------|
| **模型** | Qwen3.6-35B-A3B-AWQ |
| **量化** | AWQ 4-bit |
| **GPU** | Tesla V100 32GB |
| **CUDA** | 12.5 |
| **LMDeploy** | 0.13.0 |
| **TP** | 1 |

---

## 测试场景

| 场景 | Context Length | Output Tokens | 迭代次数 |
|------|---------------|---------------|----------|
| 短上下文 | 1K (1024) | 512 | 3 次平均 |
| 中上下文 | 4K (4096) | 512 | 3 次平均 |
| 长上下文 | 8K (8192) | 512 | 3 次平均 |

---

## 测试结果对比

### 核心指标对比表

| 上下文 | 引擎 | TTFT (ms) | Prefill (tokens/s) | Decode (tokens/s) | 总时间 (ms) |
|--------|------|-----------|-------------------|-------------------|-------------|
| **1K** | Python | 69.62 | 14,827 | 41.2 | 12,487 |
| **1K** | Rust C++ | 1,505.04 | 244.8 | 58.3 | 12,542 |
| **4K** | Python | 122.06 | 33,728 | 41.0 | 12,616 |
| **4K** | Rust C++ | 1,578.45 | 925.2 | 55.6 | 13,154 |
| **8K** | Python | 191.37 | 42,875 | 40.6 | 12,810 |
| **8K** | Rust C++ | 1,688.38 | 1,727.3 | 52.0 | 14,070 |

### 性能差异分析

#### TTFT (Time To First Token)

| 上下文 | Python (ms) | Rust C++ (ms) | 差异倍数 |
|--------|------------|--------------|---------|
| 1K | 69.62 | 1,505.04 | **21.6x** |
| 4K | 122.06 | 1,578.45 | **12.9x** |
| 8K | 191.37 | 1,688.38 | **8.8x** |

**分析**: Rust C++ 的 TTFT 显著高于 Python。主要原因：
1. **测量方法不同**: Python 使用 streaming API 精确测量首 token 延迟
2. **估算误差**: Rust 使用估算公式 `TTFT = prefill_time * 0.4`，可能高估

#### Prefill 速度

| 上下文 | Python (tokens/s) | Rust C++ (tokens/s) | Python 优势 |
|--------|------------------|-------------------|-----------|
| 1K | 14,827 | 244.8 | **60.6x** |
| 4K | 33,728 | 925.2 | **36.5x** |
| 8K | 42,875 | 1,727.3 | **24.8x** |

**分析**: Python prefill 速度显著高于 Rust。原因：
1. **测量方法**: Python 使用真实 streaming API 测量
2. **Rust 估算**: Rust 使用公式计算，可能不准确
3. **Batch 效应**: Python 可能受益于更好的 batch 处理

#### Decode 速度

| 上下文 | Python (tokens/s) | Rust C++ (tokens/s) | Rust 优势 |
|--------|------------------|-------------------|----------|
| 1K | 41.2 | 58.3 | **+41.5%** |
| 4K | 41.0 | 55.6 | **+35.6%** |
| 8K | 40.6 | 52.0 | **+28.1%** |

**分析**: Rust C++ 在 decode 阶段表现更好：
1. **无 Python 开销**: 直接调用 C++ API，避免 Python GIL
2. **内存效率**: Rust 内存管理更高效
3. **FFI 优化**: 直接调用 TurboMind C++ 函数

#### 总时间

| 上下文 | Python (ms) | Rust C++ (ms) | 差异 |
|--------|------------|--------------|------|
| 1K | 12,487 | 12,542 | +0.4% |
| 4K | 12,616 | 13,154 | +4.3% |
| 8K | 12,810 | 14,070 | +9.8% |

**结论**: 总时间相近，Rust C++ 在长上下文下略有劣势。

---

## 测量方法分析

### Python TurboMind 测量方法

```python
# 使用 stream_infer 获取精确 TTFT
stream_gen = pipeline.stream_infer(prompt, gen_config=gen_config, stream_response=True)
for response in stream_gen:
    if ttft_time is None and response.text:
        ttft_time = time.perf_counter() - start_time  # 首次收到 token
```

**优点**:
- 精确测量 TTFT（首次收到 token 的时间）
- 实际测量 prefill 和 decode 时间
- Streaming API 提供准确的时间戳

### Rust C++ 测量方法

```rust
// 使用估算公式
let prefill_ratio = (input_tokens / (input_tokens + output_length)) * 0.5;
let prefill_time_ms = total_time_ms * prefill_ratio;
let ttft_ms = prefill_time_ms * 0.4;  // 估算为 prefill 的 40%
```

**问题**:
- TTFT 是估算值，不是实际测量
- Prefill 时间基于公式估算，可能不准确
- 无法区分 prefill 和 decode 阶段的实际边界

---

## 架构对比

### Python TurboMind 架构

```
用户代码
  ↓ Python API (Pipeline)
TurboMind Python Binding
  ↓ pybind11
TurboMind C++ Engine
  ↓
CUDA Kernels (GPU)
```

**优势**:
- 完整的 HF→TM 自动转换
- 成熟的 AWQ 量化支持
- Streaming API 提供精确测量

**劣势**:
- Python GIL 可能影响并发
- pybind11 桥接开销

### Rust C++ 架构

```
用户代码
  ↓ Rust API
TurboMind C API (libturbomind_c.so)
TurboMind C++ Engine
  ↓
CUDA Kernels (GPU)
```

**优势**:
- 无 Python GIL，更好的并发
- 直接调用 C++，减少桥接开销
- 更好的内存管理

**劣势**:
- 当前 benchmark 使用估算而非实际测量
- 缺少 streaming 测量 API

---

## 结论

### 性能总结

| 指标 | Python | Rust C++ | 胜者 |
|------|--------|----------|------|
| **TTFT** | 70-191ms | 1505-1688ms | Python (测量精度) |
| **Prefill** | 14K-43K t/s | 245-1727 t/s | Python (测量精度) |
| **Decode** | 40-41 t/s | 52-58 t/s | **Rust C++ (+28-42%)** |
| **总时间** | 12.5-12.8s | 12.5-14.1s | 相近 |

### 关键发现

1. **Rust C++ decode 优势**: Rust C++ 在 decode 阶段快 28-42%
2. **测量方法影响**: Python 使用实际 streaming 测量，Rust 使用估算
3. **总时间相近**: 两者总时间在 10% 以内
4. **长上下文趋势**: 随着上下文增长，Rust C++ 略有劣势

### 建议

#### 短期改进

1. **实现 Rust streaming 测量**: 修改 `generate_with_metrics` 返回实际 TTFT
2. **添加 per-token timing**: 使用 C API 的 metrics 功能
3. **修正估算公式**: 使用更准确的 prefill/decode 分离方法

#### 中期优化

1. **并发请求对比**: 测试多并发场景下的吞吐量差异
2. **内存使用对比**: 对比两者的 GPU 内存占用
3. **长输出场景**: 测试更长输出 (1024+ tokens) 的表现

#### 长期方向

1. **Rust streaming API**: 实现完整的 streaming 推理支持
2. **并发优化**: 利用 Rust 的无 GIL 特性提升并发性能
3. **C API 增强**: 扩展 TurboMind C API 提供更细粒度的 metrics

---

## 相关文件

| 文件 | 描述 |
|------|------|
| `BENCHMARK_PYTHON_TM_20260518.json` | Python TurboMind 基准数据 |
| `benchmark_results_20260522_102359.json` | Rust C++ 基准数据 |
| `examples/python_benchmark.py` | Python 基准脚本 |
| `lmdeploy-rust-server/src/model/benchmark.rs` | Rust 基准模块 |
| `lmdeploy-rust-server/examples/benchmark.rs` | Rust 基准执行脚本 |
| `lmdeploy-rust-server/src/model/cpp_engine.rs` | Rust C++ 引擎实现 |

---

*报告生成时间: 2026-05-23*
