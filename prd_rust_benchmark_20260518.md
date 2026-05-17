# PRD: LMDeploy Rust Server Benchmark 与性能对比

**创建日期**: 2026-05-18
**状态**: Draft
**Epic ID**: `lmdeploy-bench`

---

## 背景

LMDeploy Rust Server 通过 TurboMind C API (`libturbomind_c.so`) 实现了 Qwen3.6-35B 模型推理。当前状态：
- 基础架构已实现（HTTP/gRPC 服务器、模型管理、tokenizer）
- C API FFI 绑定已完成
- 但存在运行和性能问题需要修复

Python 版本 LMDeploy 在 TurboMind 后端下预填充速度可达数千至数万 tokens/s。需要验证 Rust 版本是否能够达到同等性能。

---

## 目标

1. **修复 Rust Server 运行问题**：确保能够成功加载和运行 Qwen3.6-35B-A3B-AWQ 权重
2. **建立性能基准**：在不同 context 大小下测量 prefill (TTFT) + decode 速度
3. **与 Python 版本对比**：使用相同口径进行性能对比

---

## 用户故事

### US-001: AWQ 模型加载修复

**描述**: 作为开发者，我需要 Rust Server 能够加载 HuggingFace 格式的 AWQ 量化模型

**当前问题**:
- C API `TM_TurboMind_InitFromPath()` 不能直接加载 `.safetensors` 格式
- 需要预转换为 TurboMind `.bin` 格式
- Python 版本自动处理 HF→TM 转换

**验收标准**:
- [ ] 能够成功加载 Qwen3.6-35B-A3B-AWQ 模型
- [ ] 模型初始化无错误
- [ ] 可以执行推理请求

### US-002: 基准测试框架实现

**描述**: 作为测试工程师，我需要标准化的基准测试工具

**验收标准**:
- [ ] 支持 TTFT (Time To First Token) 测量
- [ ] 支持多种 context 长度 (1K, 2K, 4K, 8K)
- [ ] 支持 decode 速度测量
- [ ] 输出标准 JSON 格式结果

### US-003: 多场景性能测试

**描述**: 作为性能分析师，我需要不同场景下的性能数据

**测试场景**:
| 场景 | Context Length | Output Length |
|------|----------------|---------------|
| 短上下文 | 1K | 512 |
| 中上下文 | 4K | 512 |
| 长上下文 | 8K | 512 |

**验收标准**:
- [ ] 每个场景运行 3 次取平均值
- [ ] 记录 TTFT、预填充速度、解码速度
- [ ] 记录显存占用

### US-004: Python 对比基准

**描述**: 作为产品经理，我需要了解 Rust 版本与 Python 版本的性能差异

**验收标准**:
- [ ] 使用相同模型权重
- [ ] 使用相同测试场景
- [ ] 使用相同配置参数
- [ ] 生成对比报告

---

## 性能指标

### 测量方法

- **TTFT (Time To First Token)**: 从请求开始到第一个 token 生成的时间 (ms)
- **Prefill 速度** = `context_length / (TTFT / 1000)` (tokens/s)
- **Decode 速度** = `output_tokens / decode_time` (tokens/s)

### 目标

Python TurboMind 版本参考值 (Qwen3.6-35B-A3B-AWQ, V100 32GB):
- Prefill: ~数千 tokens/s
- Decode: ~40 tokens/s

---

## 技术方案

### 模型权重路径

```
/mnt/eaget-4tb/modelscope_models/tclf00/Qwen3___6-35B-A3B-AWQ
```

### 配置参数

```toml
[model]
model_path = "/mnt/eaget-4tb/modelscope_models/tclf00/Qwen3___6-35B-A3B-AWQ"
session_len = 8192
batch_size = 1
tp_size = 1
data_type = "fp16"
```

### 输出格式

```jsonc
{
  "engine": "LMDeploy Rust Server (C API)",
  "model": "Qwen3.6-35B-A3B-AWQ",
  "gpu": "Tesla V100 32GB",
  "date": "2026-05-18",
  "results": [
    {
      "scenario": "short_context",
      "context_length": 1024,
      "output_length": 512,
      "prefill_tokens": 1024,
      "ttft_ms": 123.45,
      "prefill_speed": 8296.5,
      "decode_tokens": 512,
      "decode_time_ms": 12500,
      "decode_speed": 40.96
    }
  ]
}
```

---

## 任务分解

### Epic: `lmdeploy-bench`

1. **Task**: 分析当前 AWQ 模型加载失败原因
2. **Task**: 实现模型权重转换或直接加载方案
3. **Task**: 创建基准测试框架
4. **Task**: 执行多场景性能测试
5. **Task**: 执行 Python 版本对比测试
6. **Task**: 生成性能对比报告

---

## 依赖关系

```
US-001 (模型加载) ─┐
                  ├──> US-003 (性能测试)
US-002 (测试框架) ─┘
                  │
                  └──> US-004 (Python 对比)
```

---

## 风险与限制

1. **AWQ 模型格式**: C API 可能需要预转换
2. **显存限制**: V100 32GB 可能不足以支持大 batch size
3. **量化支持**: 需要确认 AWQ 4-bit 在 C API 中的支持状态

---

## 参考资料

- [LMDeploy TurboMind C API](../../src/turbomind/capi/turbomind_c.h)
- [Python 基准测试脚本](../../scripts/benchmark_tm.py)
- [现有性能记录](../../BENCHMARK_TM_QWEN36_35B_AWQ_20260517.md)
