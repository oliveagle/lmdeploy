# LMDeploy Rust Server 纯 C++ Benchmark 报告

**报告日期**: 2026-05-21
**状态**: 🔴 **执行失败 - 权重加载错误**

---

## 1. 概述

用户要求进行 **Rust Server + C++** 的性能基准测试，**禁止 Python bridge**。

**执行结果**: ❌ **程序启动成功，权重加载阶段挂起 >180 秒后超时**

### 期望架构
```
HTTP 请求 (Axum)
  ↓ Rust Handler
TurboMind C API (libturbomind_c.so) ← 纯 C++，无 Python
  ↓
TurboMind C++ Engine
  ↓
CUDA Kernels (GPU)
```

### 当前已实现
- ✅ Rust FFI 绑定 (`src/turbomind_c.rs`)
- ✅ C++ C API (`src/turbomind/capi/turbomind_c.cc`)
- ✅ Benchmark 框架 (`src/model/benchmark.rs`)
- ✅ C++ Engine (`src/model/cpp_engine.rs`)
- ⚠️ 编译通过但**无法运行**

---

## 2. 无法运行原因分析

### 问题 1: NCCL 运行时库缺失 (阻塞等级: 🔴 高)

**错误信息**:
```
./target/debug/cpp_engine_test: error while loading shared libraries:
  libnccl.so.2: cannot open shared object file: No such file or directory
```

**根本原因**:
- `libturbomind_c.so` 依赖 NCCL (NVIDIA Collective Communications Library)
- 系统未安装 NCCL 运行时库
- CUDA 基本库 (`libcudart.so.12`, `libcuda.so.1`) 已正确链接

**验证**:
```bash
$ ldd ./target/debug/cpp_engine_test | grep "not found"
  libnccl.so.2 => not found  ← 缺失
  libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12 ✓
  libcuda.so.1 => /usr/lib/x86_64-linux-gnu/libcuda.so.1 ✓
```

**解决路径**:
```bash
# 方案 A: 从包管理器安装
sudo apt install libnccl2 libnccl-dev

# 方案 B: 从 CUDA Toolkit 复制
cp /usr/local/cuda/lib64/libnccl.so.* /path/to/nccl/lib/
export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH
```

---

### 问题 2: C++ 引擎 AWQ 权重加载未完成 (阻塞等级: 🔴 高)

**错误信息** (历史):
```
Check failed: l0 (ModelWeight::prepare 期望 layer 结构已填充)
```

**根本原因**:
- Python Pipeline 自动完成 HF→TM 格式转换 (加载到 GPU 内存)
- C++ `InitFromPath` 期望 TurboMind 原生格式的 `.bin` 文件
- **C API 的 safetensors reader 仅做基本 JSON header 解析**
- **缺少 AWQ scales/zeros 的解包和反量化逻辑**

**已创建的 Beads**:
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| `lmdeploy-85l` | 分析 C++ 引擎 AWQ 加载失败根因 | P0 | in_progress |
| `lmdeploy-5wx` | 分析 Python TurboMind 权重加载流程 | P0 | in_progress |
| `lmdeploy-82u` | 在 C++ 层构建完整 ModelWeight 层级结构 | P1 | in_progress |

---

## 3. 阻塞依赖链

```
lmdeploy-64a: 安装 NCCL 运行时库
      ↓ (blocks)
lmdeploy-109: 实现 C++ 引擎 AWQ 权重加载支持
      ↓ (blocks)
lmdeploy-0bv: 实现 Rust Server 纯 C++ benchmark 工具
```

### 当前阻塞点

| 层级 | Bead ID | 标题 | 阻塞原因 |
|------|---------|------|----------|
| 底层依赖 | `lmdeploy-64a` | 安装 NCCL | 系统库缺失 |
| 权重加载 | `lmdeploy-109` | AWQ 支持 | C++ safetensors 解析不完整 |
| Benchmark | `lmdeploy-0bv` | Benchmark 工具 | 依赖上层完成 |

---

## 4. 已有的 Python TurboMind 基准数据

### 测试环境

| 项目 | 配置 |
|------|------|
| **模型** | Qwen3.6-35B-A3B-AWQ |
| **GPU** | Tesla V100 32GB |
| **引擎** | Python TurboMind (pybind11) |

### Python 基准结果 (2026-05-18)

| Context | TTFT | Prefill Speed | Decode Speed | ITL |
|---------|------|--------------|-------------|-----|
| 1K | 69.62 ms | 14,827 t/s | 41.2 t/s | 24.3 ms |
| 4K | 122.06 ms | 33,728 t/s | 41.0 t/s | 24.45 ms |
| 8K | 191.37 ms | 42,875 t/s | 40.6 t/s | 24.69 ms |

### 预期 Rust+C++ vs Python 对比

| 指标 | Python (预期) | Rust C++ (预期) | 优势来源 |
|------|--------------|----------------|---------|
| Decode 速度 | ~41 t/s | ~41 t/s | 相同 (受 GPU 带宽限制) |
| 并发性能 | 受 GIL 影响 | 无 GIL | Rust 胜出 |
| 启动延迟 | pybind11 桥接 | 直接调用 | Rust 略优 |
| Streaming | ✅ 完整 | ⚠️ 待实现 | Python |

---

## 5. 下一步行动

### 立即行动 (解锁阻塞)

1. **安装 NCCL 运行时** (`lmdeploy-64a`)
   ```bash
   # 检查 CUDA 版本并安装对应 NCCL
   nvcc --version  # 查看 CUDA 版本
   sudo apt install libnccl2 libnccl-dev
   ```

2. **验证 C++ 引擎加载非量化模型**
   - 使用不含 AWQ 的 FP16 模型测试
   - 确认 C++ 路径完整性

### 中期行动 (完成 AWQ 支持)

3. **实现 AWQ 权重加载** (`lmdeploy-109`)
   - 分析 `ModelLoader.export()` 的转换逻辑
   - 在 C++ 层实现 safetensors → GPU 内存转换
   - 添加 AWQ scales/zeros 反量化

4. **完善 benchmark 工具** (`lmdeploy-0bv`)
   - 精确测量 TTFT (首 token 时间)
   - 分离 prefill/decode 速度
   - 并发性能测试

---

## 6. 创建的 Beads

| ID | Title | Type | Priority | Labels |
|----|-------|------|----------|--------|
| `lmdeploy-64a` | 安装 NCCL 运行时库以支持 C++ 引擎 | bug | P1 | cpp,nccl |
| `lmdeploy-109` | 实现 C++ 引擎 AWQ 权重加载支持 | feature | P0 | cpp,awq |
| `lmdeploy-0bv` | 实现 Rust Server 纯 C++ benchmark 工具 | task | P1 | cpp,benchmark |

### 依赖关系
```
lmdeploy-64a (NCCL)
    ↓ blocks
lmdeploy-109 (AWQ 支持)
    ↓ blocks
lmdeploy-0bv (Benchmark)
```

---

## 7. 结论

### 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Rust FFI 绑定 | ✅ 完成 | 可编译 |
| C++ C API | ✅ 完成 | 存在但有功能缺失 |
| Benchmark 框架 | ✅ 完成 | 代码完整 |
| NCCL 运行时 | ❌ 缺失 | 阻止运行 |
| AWQ 加载 | ❌ 不完整 | 阻止 AWQ 模型加载 |
| E2E Benchmark | ⏳ 等待 | 依赖上层解决 |

### 解决方案路径

```
1. 安装 NCCL → 2. 完成 AWQ 加载 → 3. 运行 Benchmark → 4. 对比 Python
```

---

**报告生成时间**: 2026-05-21 16:00
**相关文件**:
- `lmdeploy-rust-server/src/model/cpp_engine.rs` - C++ 引擎实现
- `lmdeploy-rust-server/src/turbomind_c.rs` - FFI 绑定
- `lmdeploy-rust-server/src/model/benchmark.rs` - Benchmark 框架