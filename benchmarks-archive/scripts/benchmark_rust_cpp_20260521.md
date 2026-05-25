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

## 2. 实际执行结果

### 2.1 NCCL 问题已解决 ✅

NCCL 存在于多个 Python venv 中，设置 `LD_LIBRARY_PATH` 后可用：

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

### 2.2 程序启动成功但权重加载挂起 🔴

```
05-21T15:43:56.085Z ✅ Initializing TurboMind C++ engine
05-21T15:43:57.290Z ✅ Tokenizer loaded successfully (vocab_size=248070)
05-21T15:43:57.299Z ✅ Creating TurboMind C++ instance...
05-21T15:43:57.311Z ⚠️ Loading weights from safetensors...
→ 180 秒后超时 (SIGKILL)
```

**问题**: 174 个 tensors 的遍历全部失败，权重加载进入死循环

### 2.3 权重加载错误详细分析

#### 错误类型 1: `linear_attn` 子模块缺失 (DeltaNet 层)

```
child('linear_attn'): (nil) (type=null)
Module tree navigation failed for 'layers.0.linear_attn.A_log'
```

**含义**: `DecoderLayerWeight` 的模块树中没有 `linear_attn` 子模块。
**原因**: C++ 构建模型层级结构时，未为 Qwen3.5 这类 DeltaNet 架构的层创建 `linear_attn` 子模块。

#### 错误类型 2: `LinearWeight` 的 `weight`/`zeros` 参数为空

```
Param 'weight' not found in module 'text_model.layers.1.feed_forward.w2' (path='layers.1.feed_forward.w2.weight', type='LinearWeight')
Param 'zeros' not found in module 'text_model.layers.1.feed_forward.w2' (path='layers.1.feed_forward.w2.zeros', type='LinearWeight')
```

**含义**: 找到了 `LinearWeight` 模块，但其 `weight` 和 `zeros` 参数没有被分配。
**原因**: `ModelWeight::prepare()` 未为 AWQ 量化模型分配参数字段，或者分配逻辑错误（与 `lmdeploy-4rt: 修复权重加载路径映射` 和 `lmdeploy-85l` 相关）。

#### 错误类型 3: 路径映射失败

```
'model.language_model.layers.0.post_attention_layernorm.weight' → 'layers.0.ffn_normm.weight'
  → Module tree navigation failed
'model.language_model.layers.3.self_attn.k_norm.weight' → 'layers.3.attention.k_norm.weight'
  → Module tree navigation failed
```

**含义**: C++ safetensors reader 的路径映射逻辑将 HF 路径转为 TM 路径，但目标模块不存在。

### 2.4 总结表

| 错误类型 | 出现次数 | 严重程度 | 关联 Beads |
|----------|---------|---------|------------|
| linear_attn 缺失 | ~100 次 | 🔴 Blocker | `lmdeploy-82u` (构建 ModelWeight) |
| Param 'weight' 为空 | ~60 次 | 🔴 Blocker | `lmdeploy-4rt` (路径映射) |
| 路径映射失败 | ~14 次 | 🔴 Blocker | `lmdeploy-4rt` (路径映射) |

---

## 3. 历史阻塞问题（已解决/部分解决）

### ~~问题: NCCL 运行时库缺失~~ ✅ **已解决**
通过设置 LD_LIBRARY_PATH 指向 Python venv 中的 NCCL 库解决。

## 4. 权重加载阻塞根因分析

### 根本原因: C++ 层 ModelWeight 层级结构不完整

C++ 引擎在 `InitFromPath` 时，通过解析 `config.json` 动态构建 `ModelWeight` 树，然后遍历 safetensors 并将每个 tensor 映射到树中的模块参数。**当前有三类错误**：

1. **线性注意力子模块缺失**: Qwen3.5 使用 Gated Delta Net (`linear_attn` + DeltaNet)，但 C++ 的 `DecoderLayerWeight` 只构建了 Attention / FFN / Norm，没有 `linear_attn` 子模块。

2. **路径映射失败**: HF 权重的路径名（如 `layers.0.post_attention_layernorm.weight`）映射到 TM 内部路径（如 `layers.0.ffn_normm.weight`）时，目标模块不存在。

3. **`LinearWeight::weight/zeros` 为空**: 找到 `LinearWeight` 模块但其参数未分配，可能与 `ModelWeight::prepare()` 中 `LinearWeight::param()` 返回空有关（`lmdeploy-4rt` P0）。

**根本解决方案**:
- 需要在 `InitFromPath` 中，根据模型架构类型（Qwen2 / Qwen3 / GatedDeltaNet），动态添加缺失的子模块节点。
- 同时需要完善路径别名映射（如 `post_attention_layernorm` → `ffn_normm`）。

---

## 5. 当前阻塞依赖链

```
lmdeploy-4rt [P0]: 修复权重加载路径映射
      ↓ blocks (param lookup)
lmdeploy-82u [P1]: 在 C++ 层构建完整 ModelWeight 层级结构 (linear_attn 子模块)
      ↓ blocks (weight loading)
lmdeploy-s8l [P1]: 测试 C++ 引擎端到端推理
      ↓ blocks (E2E)
lmdeploy-0bv [P1]: 实现 Rust Server 纯 C++ benchmark 工具
```

### 当前阻塞点

---

### 当前阻塞点

| 层级 | Bead ID | 标题 | 阻塞原因 | 状态 |
|------|---------|------|----------|------|
| 参数查找 | `lmdeploy-4rt` | 修复权重加载路径映射 | `LinearWeight::param()` 返回空 | in_progress |
| 层级构建 | `lmdeploy-82u` | 构建完整 ModelWeight 层级结构 | 缺失 `linear_attn` 子模块 | in_progress |
| AWQ 加载 | `lmdeploy-85l` | 分析 AWQ 加载失败根因 | 路径映射失败 | in_progress |
| E2E 验证 | `lmdeploy-s8l` | 测试 C++ 引擎端到端推理 | 权重加载挂起 | in_progress |
| Benchmark | `lmdeploy-0bv` | 实现纯 C++ benchmark 工具 | 依赖 E2E 验证 | open |

---

## 6. 已有的 Python TurboMind 基准数据

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

---

## 7. 已完成的部分 (历史记录)

### ✅ Rust FFI 绑定 (`lmdeploy-m9v`)
- 完整的 C API 声明 (`TM_CreateEngine`, `TM_InitFromPath`, `TM_Forward`, 等)
- Rust 包装 (`TurboMindEngine`, `TurboMindCEngine`)
- FFI 调用无 Python 依赖

### ✅ C++ C API
- 支持从 safetensors 直接加载，无需 Python 转换 (`turbomind_c.cc`)
- AWQ 量化配置 (quant_policy=4) 可设置
- Streaming 输出支持

### ✅ Benchmark 框架 (`benchmark.rs`)
- TTFT/Prefill/Decode 测量逻辑
- 多 context length (1K/4K/8K) 配置

---

## 8. 下一步行动优先级

### 立即解决 (P0)

1. **修复 `LinearWeight::param()` 返回空** (`lmdeploy-4rt`)
   - 调试 `turbomind/capi/turbomind_c.cc` 中的 `LinearWeight::param()`
   - 确保 weights/zeros 指针正确分配
2. **分析 AWQ 路径映射失败** (`lmdeploy-85l`)
   - 比较 Python vs C++ 的路径转换
   - 找出缺失别名的映射表

### 高优先级 (P1)

3. **构建 `linear_attn` 子模块** (`lmdeploy-82u`)
   - 在 `DecoderLayerWeight` 初始化中添加 `linear_attn` 分支
   - 处理 `conv1d`, `dt_bias`, `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `norm` 等参数
4. **完成 E2E 验证** (`lmdeploy-s8l`)
   - 确保权重加载在合理时间（< 30s）内完成
   - 运行单 token 生成测试

### 基准收集

5. **实现纯 C++ benchmark** (`lmdeploy-0bv`)
   - 使用 `cpp_engine_test.rs` 扩展为完整 benchmark
   - 收集 TTFT/Prefill/Decode，与 Python 对比

---

## 9. 已创建的 Beads (最新状态)

| ID | Title | Type | Priority | Labels | Status |
|----|-------|------|----------|--------|--------|
| `lmdeploy-4rt` | 修复权重加载路径映射 | bug | P0 | cpp | in_progress |
| `lmdeploy-85l` | 分析 C++ 引擎 AWQ 加载失败根因 | task | P0 | cpp | in_progress |
| `lmdeploy-82u` | 在 C++ 层构建完整 ModelWeight 层级结构 | task | P1 | cpp | in_progress |
| `lmdeploy-s8l` | 测试 C++ 引擎端到端推理 | task | P1 | cpp | in_progress |
| `lmdeploy-0bv` | 实现 Rust Server 纯 C++ benchmark 工具 | task | P1 | cpp,benchmark | open |

### 依赖关系
```
lmdeploy-4rt (路径映射) → lmdeploy-82u (层级结构) → lmdeploy-s8l (E2E) → lmdeploy-0bv (Benchmark)
```

---

## 10. 结论

### 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Rust FFI 绑定 | ✅ 完成 | 可编译，无 Python 依赖 |
| C++ C API | ✅ 完成 | 存在但需要权重加载修复 |
| NCCL 运行时 | ✅ 可用 | 通过 LD_LIBRARY_PATH 指向 venv 中的库 |
| 权重加载 | ❌ 挂起 | ModelWeight 层级结构不完整，路径映射失败 |
| E2E Benchmark | ❌ 等待 | 依赖权重加载问题解决 |

### 预期解决方案路径

```
1. lmdeploy-4rt (param 空问题)
   ↓
2. lmdeploy-82u (构建 linear_attn 层级) + lmdeploy-85l (路径映射)
   ↓
3. lmdeploy-s8l (E2E 验证)
   ↓
4. lmdeploy-0bv (Benchmark)
```

**当前 Blockers**: `lmdeploy-4rt` P0 - 需要解决 `LinearWeight::param()` 返回空才能继续。

---

**报告生成时间**: 2026-05-21 16:20
**最后更新**: 实际运行测试 2026-05-21 15:43-16:03
**相关文件**:
- `lmdeploy-rust-server/src/model/cpp_engine.rs` - C++ 引擎实现
- `lmdeploy-rust-server/src/turbomind/capi/turbomind_c.cc` - C++ C API (核心问题区域)
- `lmdeploy-rust-server/src/model/benchmark.rs` - Benchmark 框架
- `reports/benchmark_rust_cpp_20260521.md` - 本文档