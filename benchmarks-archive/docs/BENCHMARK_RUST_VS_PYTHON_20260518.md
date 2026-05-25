# LMDeploy Rust vs Python 性能对比报告

## 测试环境

| 项目 | 配置 |
|------|------|
| **模型** | Qwen3.6-35B-A3B-AWQ |
| **量化** | AWQ 4-bit |
| **GPU** | Tesla V100 32GB (PG503-216) |
| **CUDA** | 12.5 |
| **LMDeploy** | 0.13.0 |
| **测试日期** | 2026-05-18 |
| **TP** | 1 |

## 测试场景

| 场景 | Context Length | Output Tokens | 迭代次数 |
|------|---------------|---------------|----------|
| 短上下文 | 1K (1024) | 512 | 3 次平均 |
| 中上下文 | 4K (4096) | 512 | 3 次平均 |
| 长上下文 | 8K (8192) | 512 | 3 次平均 |

## 测试方法

### Python TurboMind

- **API**: `lmdeploy.Pipeline` + `TurbomindEngineConfig`
- **TTFT**: stream API 首 token 延迟精确测量
- **ITL**: Inter-token latency 通过连续 token 间隔计算
- **源码**: `examples/python_benchmark.py`
- **原始结果**: `BENCHMARK_PYTHON_TM_20260518.json`

### Rust Server

- **API**: `lmdeploy-rust-server` 直接调用 TurboMind C API
- **工具**: `BenchmarkRunner` (src/model/benchmark.rs)
- **执行**: `cargo run --example benchmark`
- **状态**: ⚠️ 工具已实现但无法执行（详见下方阻塞问题）

---

## 测试结果

### Python TurboMind 基准测试

| 指标 | 1K Context | 4K Context | 8K Context |
|------|-----------|-----------|-----------|
| **TTFT** | 69.62 ms | 122.06 ms | 191.37 ms |
| **Prefill 速度** | 14,827 tokens/s | 33,728 tokens/s | 42,875 tokens/s |
| **Decode 速度** | 41.2 tokens/s | 41.0 tokens/s | 40.6 tokens/s |
| **ITL (平均)** | 24.3 ms | 24.45 ms | 24.69 ms |
| **总时间** | 12,487 ms | 12,616 ms | 12,810 ms |

### Rust Server 基准测试

| 指标 | 1K Context | 4K Context | 8K Context |
|------|-----------|-----------|-----------|
| **状态** | ⚠️ 阻塞 | ⚠️ 阻塞 | ⚠️ 阻塞 |

---

## 阻塞问题分析

### 根本原因：C API 无法加载 HuggingFace 格式

Rust Server 通过 `libturbomind_c.so` 的 `InitFromPath` 加载模型，但 C API 的 safetensors reader 仅做基本的 JSON header 解析，**不包含** AWQ scales/zeros 的解包和反量化逻辑。

**错误信息**:
```
Check failed: l0 (ModelWeight::prepare 期望 layer 结构已填充)
```

**Python 为什么能工作**:
```
Pipeline() → TurboMind.__init__() → _from_hf() → ModelLoader.export()
```
Python 桥接自动完成 HF→TurboMind 权重转换（加载到 GPU 内存），但**不写入磁盘**。

**Rust Server 的困境**:
1. 直接 `InitFromPath` → 失败（期望 TurboMind 格式的 .bin 文件）
2. C API 的 `InitFromHF()` → 标记为 `TM_ERR_NOT_IMPLEMENTED`
3. Python 转换只加载到内存 → 无法复用

### 已尝试的解决方案

| 方案 | 状态 | 说明 |
|------|------|------|
| A. 预转换模型到 TurboMind 格式 | 理论可行 | Python 不写入磁盘，需手动导出 |
| B. Rust 自动检测并调用 Python 转换 | 已实现检测 | `engine.rs` 有 HF 检测逻辑，但转换脚本路径依赖问题 |
| C. 扩展 C API 原生支持 AWQ | 长期方案 | 需要实现 safetensors 解析 + AWQ 反量化 |

---

## Python 性能详细分析

### Prefill 速度分析

Python TurboMind 的 prefill 速度随 context 增长而提高：

```
1K:  14,827 tokens/s  ████████
4K:  33,728 tokens/s  ██████████████████
8K:  42,875 tokens/s  ████████████████████████
```

**原因**: 更大的 batch 更好地利用 GPU 并行计算资源。

### Decode 速度分析

Decode 速度几乎不随 context 变化：

```
1K:  41.2 tokens/s  ████████████████████████
4K:  41.0 tokens/s  ████████████████████████
8K:  40.6 tokens/s  ████████████████████████
```

**原因**: Decode 阶段是 memory-bound 操作，主要受限于显存带宽。

### TTFT 分析

TTFT 随 context 线性增长：

```
1K:  69.62 ms   ████████████████
4K:  122.06 ms  ████████████████████████████
8K:  191.37 ms  ████████████████████████████████████████████
```

**原因**: TTFT 包含 prefill 阶段的计算，context 越长计算量越大。

### Inter-Token Latency (ITL) 分析

ITL 几乎恒定在 ~24ms，非常稳定：

```
1K:  24.30 ms   ████████████████████████
4K:  24.45 ms   ████████████████████████
8K:  24.69 ms   ████████████████████████
```

**原因**: 与 decode 速度一致，ITL = 1 / decode_speed ≈ 24ms/token。

---

## 历史基准数据对比

### TurboMind Python API 历史数据 (2026-05-17)

| Context | Prompt Tokens | Decode Tokens | Decode Speed |
|---------|--------------|---------------|-------------|
| 128 | 35 | 128 | 41.1 tokens/s |
| 512 | 83 | 128 | 41.1 tokens/s |
| 1024 | 147 | 128 | 40.8 tokens/s |
| 2048 | 275 | 128 | 40.6 tokens/s |

### 本次测试 (2026-05-18) 与历史数据对比

| Context | 历史 Decode | 本次 Decode | 差异 |
|---------|-----------|-----------|------|
| 1K | 40.8 tokens/s | 41.2 tokens/s | +1.0% |
| 8K | - | 40.6 tokens/s | - |

**结论**: Decode 速度非常稳定，在不同测试日期和方法下差异 <2%。

---

## 架构差异

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
- 完善的内存管理和 KV cache

**劣势**:
- Python GIL 可能影响并发
- 额外的 pybind11 桥接开销

### Rust Server 架构 (计划)

```
HTTP 请求
  ↓ Axum Router
Rust Handler
  ↓ TurboMind C API (libturbomind_c.so)
TurboMind C++ Engine
  ↓
CUDA Kernels (GPU)
```

**预期优势**:
- 无 GIL，更好的并发
- 原生系统调用，减少桥接开销
- 更好的错误处理和生命周期管理

**当前问题**:
- C API 缺少 HF 格式支持
- AWQ 量化权重无法直接加载

---

## 改进建议

### 短期（解除 Rust 测试阻塞）

1. **手动预转换模型**: 使用 Python 脚本导出 TurboMind 格式到磁盘
   ```python
   from lmdeploy.turbomind import TurboMind
   tm_model = TurboMind(model_path, model_name='qwen35moe')
   # 转换后保存到 workspace 目录
   ```

2. **Rust 自动转换完善**: `engine.rs` 已有 HF 检测逻辑，完善 Python 转换脚本的路径处理

### 中期（性能优化）

3. **Rust 并发优势**: 解除阻塞后对比 Rust vs Python 的并发请求性能
4. **KV cache 优化**: 调整 `cache_max_entry_count` 减少 eviction

### 长期（架构完善）

5. **C API 原生 HF 支持**: 扩展 `turbomind_c.cc` 的 safetensors reader
6. **AWQ 反量化**: 在 C API 层实现 AWQ scales/zeros 解包逻辑

---

## 结论

### Python TurboMind 性能总结

| 指标 | 表现 |
|------|------|
| **Decode 速度** | 稳定在 ~41 tokens/s |
| **Prefill 速度** | 随 context 增长，最高 42,875 tokens/s |
| **TTFT** | 1K: 70ms → 8K: 191ms |
| **稳定性** | 不同测试间差异 <2% |

### Rust Server 状态

- 基准测试框架已完整实现
- 因 C API 格式限制无法执行实际测试
- 需要手动预转换模型或完善自动转换流程

### 下一步

1. 手动转换模型到 TurboMind 格式
2. 运行 Rust benchmark 获取数据
3. 补充完整对比分析

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `examples/python_benchmark.py` | Python 基准测试脚本 |
| `lmdeploy-rust-server/src/model/benchmark.rs` | Rust 基准测试模块 |
| `lmdeploy-rust-server/examples/benchmark.rs` | Rust 基准测试执行入口 |
| `BENCHMARK_PYTHON_TM_20260518.json` | Python 测试结果 |
| `BENCHMARK_TM_QWEN36_35B_AWQ_20260517.md` | 历史性能记录 |
