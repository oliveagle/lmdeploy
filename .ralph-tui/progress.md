# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **LMDeploy TurboMind C API 模型加载**: C API 只能加载 TurboMind 转换后的 `.bin` 格式，不能直接加载 HuggingFace `.safetensors`。Python API 自动进行 HF → TM 转换，首次加载时在 workspace 目录生成转换后的权重。Rust Server 使用 C API FFI 绑定时需要提供已转换的模型路径。
- **Python TurboMind 性能基线**: V100 32GB + Qwen3.6-35B-A3B-AWQ，Decode 稳定 ~41 t/s (ITL ~24ms)，Prefill 14K-43K t/s（随 context 长度增加）。
- **lmdeploy serve api_server 自动检测**: 加载 AWQ 量化模型时自动设置 `model_format='awq'`，无需手动指定 `--model-format` 参数。
- **基准测试方法**: 使用 streaming API 测量 TTFT（第一个 token 延迟），`python -c "import lmdeploy; print(lmdeploy.__version__)"` 验证安装。

---

## 2026-05-18 - lmdeploy-0s1
- 完成了 Rust C API 模型加载问题分析和修复方案
- 实现了 Python LMDeploy TurboMind 基准测试，涵盖 1K/4K/8K context 场景
- 生成了 Rust vs Python 性能对比报告 (RUST_VS_PYTHON_BENCHMARK_20260518.md)
- 保存了 JSON 格式基准数据 (BENCHMARK_PYTHON_TM_20260518.json)
- **关键发现**:
  - `TM_TurboMind_InitFromPath()` 仅支持 TurboMind 转换后的模型格式
  - `InitFromHF` 实现是 Python bridge hack，不适用于生产环境
  - 解决方案: 使用 Python API 生成 workspace 目录，或预转换模型
  - Rust C API 正确的初始化序列: CreateContext → CreateRoot → ProcessWeights → CreateEngine
- **Learnings:**
  - LMDeploy TurboMind C API 无法直接加载 HF safetensors，需要预转换
  - Python TurboMind 自动处理 HF → TM 转换，workspace 在模型目录内生成
  - AWQ 量化模型加载时 Python API 自动检测 format，无需手动指定
  - Decode 速度稳定 ~41 t/s，ITL ~24ms，V100 32GB 单卡

