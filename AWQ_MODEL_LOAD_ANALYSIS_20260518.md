# AWQ 模型加载失败分析报告

**日期**: 2026-05-18
**Bead ID**: lmdeploy-7n4
**状态**: 分析完成

---

## 问题总结

Rust Server 使用 TurboMind C API 无法加载 AWQ 量化模型（Qwen3.6-35B-A3B-AWQ），根本原因有以下几点：

### 1. C API `InitFromPath` 不支持直接加载 HuggingFace safetensors

**文件**: `src/turbomind/capi/turbomind_c.cc`

`TM_TurboMind_InitFromPath()` 函数虽然读取了 `config.json` 中的 AWQ 配置（`quantization_config`），但它通过简单的 `ProcessWeights` + `CreateEngine` 序列加载权重。这个路径期望的是**已经转换为 TurboMind 格式**的权重文件（`.bin`），而不是 HuggingFace 的 `.safetensors` 格式。

C API 中的 safetensors 读取代码（约第 565-819 行）是简单实现，**不完整**：
- 仅支持基本的 JSON header 解析
- 缺少 AWQ 量化权重的解包（unpack）、反量化（dequantize）逻辑
- 缺少 scales/zeros 的处理

### 2. Python API 自动完成 HF → TM 转换，但 C API 没有此机制

**文件**: `lmdeploy/turbomind/turbomind.py`, `lmdeploy/turbomind/converter.py`

Python API 的关键流程：

```
HF safetensors → get_tm_config() → WeightFormatResolver → ModelLoader → export()
                                                    ↓
                                              _process_weights()
                                                    ↓
                                              _create_engine()
```

1. `_from_hf()` 读取 HF 模型的 `config.json`
2. `converter.py` 中的 `get_tm_config()` 检测 `quantization_config.quant_method == 'awq'`
3. 构建 `AWQFormat` resolver，设置正确的 `group_size=128`
4. `ModelLoader` 从 safetensors 加载权重，应用 AWQ 格式处理
5. `export()` 将权重写入 TurboMind 格式

C API 直接调用 `InitFromPath()` **跳过了所有这些转换步骤**。

### 3. Rust Engine 配置缺少 AWQ 相关参数

**文件**: `lmdeploy-rust-server/src/model/engine.rs`

Rust 代码创建 `EngineConfig` 时：
```rust
config.set_data_type(tm::TM_DataType::TM_DATATYPE_FP16);
config.set_session_len(self.session_len);
// ... 其他参数
```

**缺少**:
- `quant_policy` - AWQ 模型需要设置量化策略
- 缺少对 `model_format=awq` 的检测和传递

---

## 根本原因

| 原因 | 说明 |
|------|------|
| **格式不匹配** | C API 期望 TurboMind `.bin` 格式，但提供了 HuggingFace safetensors |
| **缺少转换** | C API 没有 Python API 那样的自动 HF→TM 转换逻辑 |
| **缺少量化处理** | C API 的 ProcessWeights 不包含 AWQ scales/zeros 的解包和反量化 |
| **配置不完整** | Rust EngineConfig 缺少 quant_policy 等 AWQ 相关参数 |

---

## 修复方案

### 方案 A: 预转换模型（推荐，最稳定）

使用 Python API 预转换 AWQ 模型到 TurboMind 格式：

```bash
python3 -c "
from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig

model_path = '/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ'
cfg = TurbomindEngineConfig(session_len=4096, max_batch_size=8)
tm = TurboMind(model_path, engine_config=cfg, trust_remote_code=True)
# 首次加载会自动完成 HF→TM 转换
print('转换完成')
"
```

然后 Rust Server 使用转换后的目录（workspace）作为模型路径。

### 方案 B: Rust Server 先调用 Python 转换

在 Rust Server 启动时，自动检测 safetensors 格式并调用 Python 脚本转换：

```rust
// 检测模型是否为 HF 格式
if has_safetensors(&model_path) {
    convert_hf_to_turbomind(&model_path)?;
}
// 然后使用转换后的路径初始化 C API
```

### 方案 C: 扩展 C API 支持 AWQ safetensors（长期）

需要在 `turbomind_c.cc` 中实现：
1. 完整的 safetensors 解析器（替代当前的简单 JSON 解析）
2. AWQ 权重的解包逻辑（参考 `weight_format.py` 中的 `AWQFormat`）
3. Scales/zeros 的加载和反量化
4. 配置系统中添加 AWQ 量化参数的传递

这需要大量 C++ 开发工作，涉及 TurboMind 内部的量化内核。

---

## 推荐方案

**短期**: 方案 A - 预转换模型
- 零代码改动
- 利用已有的 Python 转换逻辑
- 稳定可靠

**中期**: 方案 B - Rust 自动检测转换
- Rust Server 启动时自动处理
- 对用户透明

**长期**: 方案 C - C API 原生支持
- 完整支持 AWQ safetensors
- 消除 Python 依赖
- 开发成本高

---

## 当前代码状态

- `engine.rs` 中的 `init()` 方法检查 `config.yaml` / `config.json` 是否存在，但即使存在也无法正确加载 safetensors
- `turbomind_c.rs` 中的 `TurboMind::create()` → `init_from_path()` 流程缺少 AWQ 处理
- C API 中的 `TM_TurboMind_InitFromHF()` 标记为 `TM_ERR_NOT_IMPLEMENTED`
- 进度日志中已记录此问题（`.ralph-tui/progress.md`）
