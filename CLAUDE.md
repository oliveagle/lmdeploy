# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Linting:**

```bash
pre-commit run --all-files
```

Style: PEP8, max line length 120, double quotes, LF endings. C++ source under `src/` uses clang-format.

**Tests:**

```bash
pytest tests/test_lmdeploy                          # all unit tests
pytest tests/test_lmdeploy/test_model.py            # specific file
pytest tests/test_lmdeploy/test_lite/               # quantization tests
pytest tests/test_lmdeploy/test_vl/                 # vision-language tests
```

**Debug logging:**

```bash
LMDEPLOY_LOG_LEVEL=DEBUG python ...
```

**Build (TurboMind C++ extension):**

- Controlled via `setup.py` + CMake. Relevant env vars: `LMDEPLOY_TARGET_DEVICE` (default `cuda`), `DISABLE_TURBOMIND`, `CMAKE_BUILD_TYPE`, `CUDACXX`.
- Requirements split by device: `requirements/runtime_cuda.txt`, `runtime_ascend.txt`, etc.

## Architecture

### Two Backends, One Pipeline

`lmdeploy/pipeline.py` is the main user-facing entry point (`pipeline()` in `api.py`). It instantiates either the **PyTorch engine** (`lmdeploy/pytorch/`) or the **TurboMind engine** (`lmdeploy/turbomind/`) based on config.

### PyTorch Backend

**Model patching** is the core mechanism: HuggingFace models are loaded normally, then their layers are dynamically replaced with optimized LMDeploy implementations.

- `lmdeploy/pytorch/models/module_map.py` — registry mapping HF class names → LMDeploy replacement classes. Device-specific overrides in `DEVICE_SPECIAL_MODULE_MAP`.
- `lmdeploy/pytorch/models/patch.py` — applies the substitutions at runtime via `_get_rewrite_qualname()` / `_class_from_qualname()`.
- `lmdeploy/pytorch/models/` — 40+ per-model files (e.g., `llama.py`, `qwen.py`, `deepseek_v2.py`). Each reimplements attention, MLP, and embeddings using custom kernels.
- `lmdeploy/pytorch/nn/` — reusable optimized modules: `linear/` (AWQ, W8A8, blocked-FP8, LoRA variants), `attention.py`, `norm.py`, `rotary_embedding.py`, `moe/`.
- `lmdeploy/pytorch/kernels/` — Triton/CUDA kernels (e.g., `w8a8_triton_kernels.py`).
- `lmdeploy/pytorch/backends/` — kernel/operator dispatchers per quantization type (FP8, AWQ, CUDA).

**Engine execution flow (key files):**

- `engine.py` — main PyTorch engine.
- `paging/scheduler.py` — sequences → batches; prefill/decode, block eviction, prefix caching (`BlockTrie`).
- `engine/engine_loop.py` — async inference loop.
- (See `pytorch/engine/` and `pytorch/paging/` for full execution detail.)

**Configuration dataclasses** (`lmdeploy/pytorch/config.py`): `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `BackendConfig`, `DistConfig`, `MiscConfig`.

### TurboMind Backend

- Python wrapper: `lmdeploy/turbomind/turbomind.py` (~800 lines). Bridges into `lmdeploy/lib/_turbomind` (pybind11 extension built from `src/turbomind/`).
- Tensor interop via `torch.from_dlpack()` / `_tm.from_dlpack()`.
- Config and model conversion: `lmdeploy/turbomind/deploy/config.py`, `supported_models.py`.
- Parallel config helpers: `update_parallel_config()`, `complete_parallel_config()` in `messages.py`.

### Lite / Quantization

Entrypoints in `lmdeploy/lite/apis/`: `calibrate.py` (main), `auto_awq.py`, `gptq.py`, `smooth_quant.py`.

**Flow:** load HF model → `CalibrationContext` collects activation statistics → scale computation (`lmdeploy/lite/quantization/`) → write quantized weights.

- `lite/quantization/awq.py` — AWQ (NORM_FCS_MAP, FC_FCS_MAP define per-model layer structure).
- `lite/quantization/weight/quantizer.py` — weight quantizer.
- `lite/quantization/activation/observer.py` — activation statistics.
- `lite/modeling/` — model-specific GPTQ implementations (e.g., `internlm2_gptq.py`).
- `lite/utils/cal_qparams.py` — quantization parameter calculation utilities.

Layer/norm/head mappings per model family are defined directly in `calibrate.py` and `awq.py`.

### Vision-Language Models

- `lmdeploy/vl/model/` — VLM preprocessing (InternVL, Qwen-VL, LLaVA, CogVLM, etc.).
- `lmdeploy/vl/media/` — image/video loaders and base classes.
- `lmdeploy/pytorch/multimodal/` — multimodal input handling for the PyTorch engine.
- Reference VLM implementation: `lmdeploy/vl/model/qwen3.py`.

### Other Key Files

- `lmdeploy/messages.py` — core types: `GenerationConfig`, `EngineConfig`, `TurbomindEngineConfig`, `SchedulerSequence`, `MessageStatus`.
- `lmdeploy/model.py` — chat templates; critical for correct conversation formatting.
- `lmdeploy/archs.py` — architecture registry mapping model arch names to runtime patches.
- `lmdeploy/tokenizer.py` — HuggingFace/SentencePiece tokenizer wrapper.
- `lmdeploy/serve/openai/` — OpenAI-compatible API server.

## Adding a New PyTorch Model

Use the `/support-new-model` skill for a complete step-by-step guide.

---

## DFlash Speculative Decoding Integration

**当前状态**: 设计完成，C++ 原型 80%，核心集成进行中

**团队协作**: 使用多 Agent 团队 `dflash-turbomind-integration` 进行开发

**关键文档**:
- `DFLASH_PROJECT_SUMMARY.md` - 项目总结
- `DFLASH_IMPLEMENTATION_ROADMAP.md` - 实施路线图
- `DFLASH_TURBOMIND_DESIGN.md` - 架构设计
- `~/.claude/teams/dflash-turbomind-integration/AGENTS.md` - 团队协作记录

**已创建的 C++ 文件**:
- `src/turbomind/models/llama/DFlashDraftModel.{h,cc}` - Draft model 实现
- `src/turbomind/models/llama/DFlashDraftWeight.{h,cc}` - 权重结构
- `src/turbomind/models/llama/dflash_kernels.{h,cu}` - CUDA kernels
- `src/turbomind/models/llama/unified_decoder_dflash.{h,cc}` - Decoder 扩展

**待完成的任务**:
1. 修改 `LlamaWeight.{h,cc}` 添加 DFlashDraftWeight 支持
2. 修改 `LlamaDecoder.cc` 实现 speculative 解码
3. 添加 `SpeculativeConfig` 到 `messages.py`
4. 修改 `CMakeLists.txt` 添加 DFlash 源文件
5. 修改 `turbomind.py` Python 接口

**预期性能**: 1.7x+ decode speedup, 60-80% accept rate

## Qwen3.6-35B-A3B-AWQ MoE 模型支持

**当前状态**: ✅ AWQ MoE 量化方法支持已修复

**修复的问题**:
1. ✅ 添加 AWQ 量化方法到 `lmdeploy/pytorch/nn/moe/__init__.py`
2. ✅ 修复 `modules_to_not_convert` 层的识别（`lmdeploy/pytorch/config.py`）
3. ✅ 添加 EP fallback 到 Triton 实现（`lmdeploy/pytorch/backends/cuda/moe/default.py`）

**内存需求分析**:
- Qwen3.6-35B-A3B-AWQ 有 256 个专家 × 40 层
- TP4 只分片 FFN 维度，不分片专家
- MoE 权重每 GPU 需要 ~15 GB
- 总需求约 18 GB/GPU（超过 16GB V100）

**运行建议**:
- 使用 A100 40GB 或更大显存的 GPU
- 或使用 8 张 GPU（TP=8）
- 或测试更小的 MoE 模型

**已知限制**:
- 当前 LMDeploy MoE 实现 EP+TP 不能有效减少内存
- EP 只分片专家，TP 只分片 FFN，不同时分片两者
- 需要 `deep_gemm` 或 `DeepEP` 库才能使用完整的 EP 功能

## Test 文件存放规范

**测试文件位置**: `tests/` 目录下按功能分类组织

### 目录结构

```
tests/
├── dflash/              # DFlash speculative decoding 测试
│   ├── test_dflash.py
│   ├── test_dflash_debug.py
│   └── ...
├── test_awq_moe_fix.py  # AWQ MoE 相关测试
├── test_correct_cache.py
├── test_ep_config.py
└── ...
```

### 测试文件命名规范

- **文件名**: 必须以 `test_` 开头，使用下划线命名
- **位置**: 必须放在 `tests/` 目录下，按功能分子目录（如 `dflash/`）
- **禁止**: 根目录（`lmdeploy/`）下禁止放置 `test*.py` 文件

### 测试文件编写规范

1. **禁止硬编码路径**
   - ❌ 禁止: `sys.path.insert(0, '/home/oliveagle/opt/lmdeploy/lmdeploy')`
   - ✅ 正确: 假设从 repo 根目录运行，lmdeploy 已在 PYTHONPATH 中

2. **禁止硬编码 LD_LIBRARY_PATH**
   - ❌ 禁止: `os.environ['LD_LIBRARY_PATH'] = '.../build/lib'`
   - ✅ 正确: 使用 `pip install -e .` 安装后，库路径自动配置

3. **模型路径配置**
   - 使用环境变量或配置文件
   - 提供默认值示例，但允许用户覆盖

4. **GPU 配置**
   - 使用 `CUDA_VISIBLE_DEVICES` 环境变量
   - 提供清晰的注释说明 GPU 需求

### 运行测试

从 repo 根目录运行：

```bash
# 运行单个测试
python tests/dflash/test_dflash.py

# 运行所有单元测试
pytest tests/test_lmdeploy/

# 运行特定模块测试
pytest tests/test_lmdeploy/test_model.py
```

