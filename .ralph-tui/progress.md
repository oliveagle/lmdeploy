# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## 2026-05-17 - lmdeploy-jog.1.5 - AWQ 4-bit 量化参数支持

### 实现内容
1. **`EngineConfig::set_quant_policy()` 方法** - 在 `turbomind_c.rs` 中添加了设置量化策略的方法
2. **`AwqConfig` 结构体** - 匹配 C++ `AwqQuantConfig` 的 Rust 版本，包含:
   - `bits`: 量化位数 (默认 4)
   - `group_size`: 分组大小 (默认 128)
   - `version`: AWQ 版本 (默认 "gemm")
   - `symmetric`: 对称量化 (默认 true)
   - `zero_point`: 零点量化 (默认 true)
   - `pack`: 权重打包 (默认 true)
3. **单元测试** - 添加了 5 个新测试验证 AWQ 参数处理

### 文件变更
- `lmdeploy-rust-server/src/turbomind_c.rs`:
  - 添加 `set_quant_policy()` 方法
  - 添加 `AwqConfig` 结构体及其实现
  - 添加 5 个单元测试

### 关键发现

**量化策略值 (quant_policy) 的含义:**
- `0` = NONE (无量化)
- `4` = AWQ 4-bit 量化
- `8` = KV Cache INT8 量化

**C++ 端已有的 AWQ 支持:**
- `turbomind_c.cc` 中已有 `AwqQuantConfig` 结构体和 `ReadAwqQuantConfig()` 函数
- `TM_TurboMind_InitFromPath()` 会自动从 `config.json` 读取 `quantization_config` 段
- 支持的参数: `bits`, `group_size`, `version`, `symmetric`, `zero_point`, `pack`

**配置流程:**
1. C++ 侧在 `InitFromPath` 中解析 `config.json`
2. Rust 侧通过 `EngineConfig::set_quant_policy()` 设置策略
3. 量化参数影响 ModelWeight 的 `data_type` 设置

---
