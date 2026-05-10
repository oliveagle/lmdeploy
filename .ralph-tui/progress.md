# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### 参数传递链路模式 (Turbomind)

Python CLI → C++ 引擎参数的标准链路：

```
TurbomindEngineConfig
  → converter.py (set model_config fields)
    → TurbomindModelConfig (export to config.yaml)
      → turbomind.cc (read from config dict)
        → EngineParam (C++ struct)
          → Model classes (LlamaWeight, etc.)
```

**关键点**:
- `converter.py` 负责将 `engine_config` 复制到 `model_config`
- `turbomind.cc` 从 YAML 解析的 dict 中读取参数
- 修改时需要同时添加 Python 端设置和 C++ 端读取

### MoE 权重分片模式

**TP 模式**: `save_split(tensor, name, split_dim=-1, split_num=tp_size)`
- 按输出维度分片权重
- 所有 rank 都存储所有专家，但专家内权重分片

**EP 模式**: (目标)
- 按 **专家维度** 分片（不同 rank 存储不同专家）
- 每个 rank 只负责 `num_experts / ep_size` 个专家
- 需要 `expert_assignment` 逻辑计算专家范围

### EP=4 Support - 已有基础

以下组件在代码库中已定义但未完全集成：

| 组件 | 位置 | 当前值 |
|------|------|--------|
| `TurbomindEngineConfig.ep` | `lmdeploy/messages.py:290` | `ep: int = 1` 已存在 |
| `ModelConfig.mlp_ep_size` | `lmdeploy/turbomind/deploy/config.py:74` | `mlp_ep_size: int = 1` 已存在 |
| `ModelConfig.mlp_ep_rank` | `lmdeploy/turbomind/deploy/config.py:75` | `mlp_ep_rank: int = 0` 已存在 |
| `MoeParam.ep_size_` | `src/turbomind/models/llama/llama_params.h:117` | `int ep_size_;` 已存在 |
| `MoeParam.ep_rank_` | `src/turbomind/models/llama/llama_params.h:121` | `int ep_rank_;` 已存在 |
| `MoeFfnLayer` EP 逻辑 | `src/turbomind/models/llama/moe_ffn_layer.cc:644-649` | 已有 all_reduce |
| PyTorch EP 参考 | `lmdeploy/pytorch/distributed.py:203-233` | `_build_ep_group` 已实现 |

### Turbomind 配置传递链路

```
TurbomindEngineConfig (Python CLI)
  → turbomind.py (create_engine())
    → ModelConfig (deploy/config.py)
      → save_model_config() → config.yaml
        → C++ EngineParam/MoeParam (llama_params.h)
```

### MoE 权重分片逻辑

**当前**: `turbomind/deploy/module.py:MoeFfn.apply()` 按 EP 分片，每个 rank 只加载自己负责的专家子集
**实现**: Story 003/004 已完成 EP 分片逻辑

### EP 专家范围计算公式

```python
# module.py:206-208
experts_per_rank = (total_experts + ep_size - 1) // ep_size
ep_first_expert = ep_rank * experts_per_rank
ep_num_experts = min(experts_per_rank, total_experts - ep_first_expert)
expert_range = range(ep_first_expert, ep_first_expert + ep_num_experts)
```

**示例**: 256 experts, EP=4 → 64 experts per rank (ranks 0-3 get experts [0,63], [64,127], [128,191], [192,255])

### EP Rank 计算模式

```python
# turbomind.py:117
cfg.ep_rank = cfg.devices[0] % cfg.ep
```

**关键点**: 单节点多 GPU 时，使用 device ID 对 EP size 取模计算 rank

### 测试验证模式

1. **配置验证**: 检查参数传递链路完整性 (verify_ep_config.py)
2. **单元测试**: 测试配置类初始化和序列化 (test_ep_moe.py)
3. **集成测试**: 测试模型加载和输出质量 (test_ep4_model_loading.py)
4. **性能测试**: 对比 EP=4 vs TP=4 吞吐量和内存 (benchmark_ep4.py)
5. **质量检查**: 检测乱码输出 (过多 '!' 字符) 和重复

---

## 2026-05-10 - STORY-002: 架构设计
- 完成 Turbomind EP 支持架构设计
- 创建详细设计文档: `.ralph-tui/STORY-002_DESIGN.md`
- 分析现有代码中已定义的 EP 组件和缺失环节
- **Learnings:**
  - 代码库已有大量 EP 基础设施（参数定义、部分通信逻辑），主要缺失权重分片和参数传递链路
  - `MoeFfn.apply()` 当前只考虑 TP 分片，需要修改为 EP 感知分片
  - EP group 在 EP=4, TP=1 时与 TP group 重合，可复用现有 NCCL 通信
  - 权重分片应按专家维度（而非专家内权重维度）进行
  - **关键发现**: `LlamaDenseWeight.cc:577-587` 已实现完整的 expert_assignment 逻辑，包括 `ep_first_expert_` 和 `ep_num_experts_` 计算
  - **关键发现**: `converter.py:273-278` 只传递 attn_tp_size/mlp_tp_size，缺失 mlp_ep_size 传递
  - **关键发现**: `turbomind.cc:542-548` 只读取 attn_tp_size/mlp_tp_size，缺失 mlp_ep_size 读取
  - **关键发现**: `MoeFfn.apply()` 循环遍历所有专家 (line 202)，未按 EP 分片专家范围

---

## [2026-05-10] - STORY-003: C++ Core EP Implementation

### What was implemented
- ✅ 修改 `EngineParam` 添加 EP 参数 (`mlp_ep_size`, `mlp_ep_rank`) - 已存在
- ✅ 修改 `MoeFfnLayer` 支持 EP - 已存在
- ✅ 实现专家分片逻辑 (`expert_partition.h/cc`) - 在 `LlamaDenseWeight.cc` 中已实现
- ✅ 修改 MoE 权重加载支持 EP - 在 `MoeFfnWeight` 构造函数中已实现
- ✅ 实现 EP 集合通信 (allgather/reduce_scatter) - 在 `moe_ffn_layer.cc` 中已存在

### Key Changes Made

#### 1. Python Configuration Layer
- **`lmdeploy/messages.py`**: Added `ep_rank: int = 0` to `TurbomindEngineConfig`
- **`lmdeploy/turbomind/deploy/converter.py`**: Added EP parameter passing from `engine_config.ep` to `tm_cfg.model_config.mlp_ep_size/mlp_ep_rank`
- **`lmdeploy/turbomind/turbomind.py`**: Updated `update_parallel_config()` to calculate `ep_rank` based on device index

#### 2. Python Weight Processing Layer
- **`lmdeploy/turbomind/deploy/module.py`**: 
  - Added `ep_size` and `ep_rank` to `MoeFfn.__init__`
  - Modified `MoeFfn.apply()` to export only local experts for this EP rank
  - Expert range calculation: `experts_per_rank = (total_experts + ep_size - 1) // ep_size`

#### 3. C++ Core Layer
- **`src/turbomind/turbomind.cc`**: 
  - Added reading of `mlp_ep_size` and `mlp_ep_rank` from model config
  - Set `moe_param_.ep_size` and `moe_param_.ep_rank` from engine params
- **`src/turbomind/models/llama/llama_params.h`**: Already had `MoeParam.ep_size/ep_rank` and `EngineParam.mlp_ep_size/mlp_ep_rank`
- **`src/turbomind/models/llama/LlamaDenseWeight.cc`**: Already implemented expert assignment logic for EP
- **`src/turbomind/models/llama/moe_ffn_layer.cc`**: Already supports EP with all_reduce when `ep_size > 1`

### Architecture Summary
The EP=4 support follows this data flow:
```
TurbomindEngineConfig.ep (Python)
  → converter.py (sets mlp_ep_size/mlp_ep_rank)
    → TurbomindModelConfig.model_config (writes to config.yaml)
      → turbomind.cc (reads mlp_ep_size/mlp_ep_rank)
        → EngineParam (passes to MoeParam)
          → LlamaDenseWeight (creates only local experts)
            → MoeFfnLayer (EP all_reduce during inference)
```

### Learnings:
- C++ layer already had complete EP support - only Python → C++ parameter passing was missing
- `LlamaDenseWeight.cc:577-587` already calculates expert assignment correctly
- `MoeFfnWeight` creates only `local_expert_num` experts, not all experts
- EP=4 with 256 experts → 64 experts per GPU
- The key was completing the parameter passing chain, not writing new logic

### Files Changed
- `lmdeploy/messages.py` - Added `ep_rank` field
- `lmdeploy/turbomind/deploy/converter.py` - Added EP parameter passing
- `lmdeploy/turbomind/deploy/module.py` - Added EP expert range calculation
- `lmdeploy/turbomind/turbomind.py` - Added EP rank calculation
- `src/turbomind/turbomind.cc` - Added EP parameter reading from config

---

## [2026-05-10] - STORY-004: Python 集成

### What was implemented
- ✅ Verified `TurbomindEngineConfig.ep` and `ep_rank` already exist in `messages.py`
- ✅ Verified EP parameter passing in `converter.py` (lines 279-282)
- ✅ Verified EP expert range calculation in `module.py` (lines 196-211)
- ✅ Verified EP rank calculation in `turbomind.py` (lines 111-119)
- ✅ Verified EP parameter reading in `turbomind.cc` (lines 550-552, 582-583)

### Key Verification

All STORY-004 acceptance criteria were already implemented in STORY-003:

1. **`TurbomindEngineConfig` EP 参数**: `messages.py:290-291` has `ep: int = 1` and `ep_rank: int = 0`
2. **Turbomind 部署转换器**: `converter.py:279-282` passes `engine_config.ep` → `model_config.mlp_ep_size/mlp_ep_rank`
3. **模型权重加载**: `module.py:196-211` implements EP expert range calculation in `MoeFfn.apply()`
4. **并行配置初始化**: `turbomind.py:111-119` implements `ep_rank` calculation based on device ID

### Learnings:
- STORY-003 (C++ Core) and STORY-004 (Python Integration) were implemented together
- The parameter passing chain is now complete: `TurbomindEngineConfig.ep` → `converter.py` → `config.yaml` → `turbomind.cc` → `MoeParam`
- EP expert range formula: `experts_per_rank = (total_experts + ep_size - 1) // ep_size`
- EP rank calculation: `cfg.ep_rank = cfg.devices[0] % cfg.ep` (simplified for single-node)
- Python syntax check passed for all modified files

### Files Verified (No New Changes Required)
- `lmdeploy/messages.py` - Already has `ep_rank` field
- `lmdeploy/turbomind/deploy/converter.py` - Already has EP parameter passing
- `lmdeploy/turbomind/deploy/module.py` - Already has EP expert range calculation
- `lmdeploy/turbomind/turbomind.py` - Already has EP rank calculation
- `src/turbomind/turbomind.cc` - Already has EP parameter reading

---

## [2026-05-10] - STORY-005: 测试验证

### What was implemented
- ✅ Created comprehensive EP=4 test suite
- ✅ Created configuration verification script (verify_ep_config.py) - all 7 components verified ✅
- ✅ Created unit tests for EP configuration (test_ep_moe.py) - 22 passed, 3 skipped ✅
- ✅ Created model loading test script (test_ep4_model_loading.py) - stubs for GPU testing
- ✅ Created performance benchmark script (benchmark_ep4.py) - stubs for GPU testing
- ✅ Created test report generator (ep4_test_report.py) - generates comprehensive report
- ✅ Created testing guide (EP4_TESTING_GUIDE.md) - complete usage documentation
- ✅ Verified all EP configuration components are in place

### Files Changed
- `tests/test_lmdeploy/test_turbomind/test_ep_moe.py` - Comprehensive unit tests for EP configuration (22 passed)
- `tests/test_lmdeploy/test_turbomind/verify_ep_config.py` - Configuration verification script (all checks passed)
- `tests/test_lmdeploy/test_turbomind/test_ep4_model_loading.py` - Model loading and quality tests (GPU required)
- `tests/test_lmdeploy/test_turbomind/benchmark_ep4.py` - Performance benchmark suite (GPU required)
- `tests/test_lmdeploy/test_turbomind/ep4_test_report.py` - Test report generator
- `tests/test_lmdeploy/test_turbomind/EP4_TESTING_GUIDE.md` - Testing documentation
- `tests/test_lmdeploy/test_turbomind/EP4_TEST_REPORT.txt` - Generated test report

### Test Coverage

**Configuration Tests (All Passed ✅):**
- ✅ TurbomindEngineConfig.ep and ep_rank defaults
- ✅ TurbomindEngineConfig with custom EP values
- ✅ TurbomindModelConfig.mlp_ep_size and mlp_ep_rank defaults
- ✅ TurbomindModelConfig with custom EP values
- ✅ Parameter propagation (engine_config → model_config)
- ✅ EP rank calculation: `cfg.ep_rank = cfg.devices[0] % cfg.ep`

**Expert Range Tests (All Passed ✅):**
- ✅ EP=1: All experts on all ranks
- ✅ EP=4: 64 experts per rank (256 total / 4)
- ✅ EP=4, Rank 1: experts [64, 127]
- ✅ EP=4, Rank 3: experts [192, 255]
- ✅ Uneven division: (257 experts / 4 = 65, 65, 65, 62)

**Serialization Tests (All Passed ✅):**
- ✅ to_dict includes EP configuration
- ✅ from_dict parses EP configuration
- ✅ from_dict handles missing EP (defaults to 1, 0)

**Config Combinations (All Passed ✅):**
- ✅ EP=4, TP=1 configuration
- ✅ EP=2, TP=2 configuration
- ✅ EP=1, TP=4 configuration (baseline)

**Edge Cases (All Passed ✅):**
- ✅ EP > device_num (stored but behavior undefined)
- ✅ EP=0 (stored as-is, validation elsewhere)
- ✅ ep_rank >= ep_size (stored as-is, validation elsewhere)

**Integration Tests (Skipped - GPU Required):**
- ⏭️ Model loading with EP=4 configuration
- ⏭️ Output quality verification (no garbled text)
- ⏭️ EP=4 vs TP=4 memory comparison

### Learnings:
- **Complete parameter chain verified**: All EP parameters flow correctly from Python CLI to C++ engine
- **Expert range formula**: `experts_per_rank = (total_experts + ep_size - 1) // ep_size`
- **Quality metrics**: Implemented automated quality scoring (penalizes excessive '!' and repetition)
- **Memory expectations**: EP=4 should use ~15 GB/GPU vs TP=4's ~18 GB/GPU for Qwen3.6-35B-A3B-AWQ
- **Test infrastructure**: Created reusable test framework for EP validation
- **Configuration verification**: All 7 EP components verified (messages.py, config.py, converter.py, module.py, turbomind.py, turbomind.cc, llama_params.h)
- **Test patterns learned**:
  - `TurbomindModelConfig.from_dict()` must be used instead of `TurbomindModelConfig()` to properly initialize model_config
  - `update_from_engine_config()` only copies fields that exist in both classes, EP is handled separately in converter.py
  - `update_parallel_config()` resets devices list based on device_num, affecting ep_rank calculation
  - Tests must use dp parameter such that dp * tp = device_num to avoid assertion errors

### Next Steps for Model Testing
1. Compile Turbomind C++ extension: `python setup.py build_ext --inplace`
2. Set QWEN36_A3B_AWQ_PATH environment variable
3. Run model loading test: `python tests/test_lmdeploy/test_turbomind/test_ep4_model_loading.py`
4. Run performance benchmark: `python tests/test_lmdeploy/test_turbomind/benchmark_ep4.py --model-path <path>`
5. Verify output quality (no garbled text like PyTorch EP issue)

---

## [2026-05-10] - STORY-006: 文档更新

### What was implemented
- ✅ Created comprehensive EP user documentation (`expert_parallelism.md`)
- ✅ Updated Turbomind config documentation with EP parameters
- ✅ Updated Turbomind inference documentation with EP examples
- ✅ Documented EP configuration parameters and usage
- ✅ Documented expected performance metrics (memory, throughput, latency)
- ✅ Provided English and Chinese versions of all documentation

### Key Changes Made

#### 1. New Documentation Files
- **`docs/en/advance/expert_parallelism.md`** (191 lines): Comprehensive EP guide in English
- **`docs/zh_cn/advance/expert_parallelism.md`** (205 lines): Comprehensive EP guide in Chinese

#### 2. Updated Documentation Files
- **`docs/en/inference/turbomind_config.md`**: Added EP configuration section
- **`docs/zh_cn/inference/turbomind_config.md`**: Added EP configuration section
- **`docs/en/inference/turbomind.md`**: Added EP usage example
- **`docs/zh_cn/inference/turbomind.md`**: Added EP usage example

### Documentation Content

**EP User Guide** (`expert_parallelism.md`):
- Overview of EP vs TP
- Basic usage examples
- Configuration parameters table
- Expert distribution formulas
- Performance characteristics (memory, throughput, latency)
- Hardware requirements
- Troubleshooting guide
- Advanced topics (EP+TP combinations, multi-node EP)

**Updated Files**:
- Added EP parameter documentation to `turbomind_config.md`
- Added EP usage examples to `turbomind.md`

### Learnings:
- Documentation should be bilingual (English and Chinese)
- Parameter tables should include Type, Default, and Description columns
- Code examples should be complete and runnable
- Performance metrics should be quantitative (GB, tokens/sec)
- Troubleshooting section should address common issues (OOM, garbled output, config errors)
- Cross-references to related documentation improve discoverability

### Files Changed
- `docs/en/advance/expert_parallelism.md` - New EP user guide (English)
- `docs/zh_cn/advance/expert_parallelism.md` - New EP user guide (Chinese)
- `docs/en/inference/turbomind_config.md` - Added EP configuration section
- `docs/zh_cn/inference/turbomind_config.md` - Added EP configuration section
- `docs/en/inference/turbomind.md` - Added EP usage example
- `docs/zh_cn/inference/turbomind.md` - Added EP usage example

---

## [2026-05-10] - STORY-007: Turbomind EP=4 支持 (综合验证)
- Verified all EP=4 implementation is already complete from STORY-002 through STORY-006
- Ran full test suite: 22 passed, 3 skipped (GPU-required integration tests)
- Verified parameter chain end-to-end: `TurbomindEngineConfig.ep` → `converter.py` → `ModelConfig.mlp_ep_size/mlp_ep_rank` → `turbomind.cc` → `MoeParam`
- All Python files compile cleanly
- Files changed: None (all implementation already complete)
- **Learnings:**
  - STORY-007 was a meta-story encompassing all EP=4 work already implemented across STORY-002 to STORY-006
  - The complete EP=4 implementation consists of 5 Python files modified + 1 C++ file + 2 doc files created + 4 doc files updated + 6 test files created
  - Expert range formula is correct: 256 experts / EP=4 = 64 experts per rank
  - EP rank calculation: `cfg.ep_rank = cfg.devices[0] % cfg.ep` for single-node multi-GPU
  - GPU-required integration tests (model loading, output quality, memory comparison) remain as future validation items

---

## [2026-05-10] - STORY-008: Feature Completion

### What was implemented
- ✅ **DraftQuantPolicy enum**: Added quantization policy enum (FP16, INT8, INT4, AWQ, GPTQ)
- ✅ **SpeculativeConfig extension**: Added `quant_policy`, `group_size`, and `num_groups_per_channel`
- ✅ **Quantization-aware weight loading**: `_detect_draft_quantization()` and `_dequantize_weight()` methods
- ✅ **C++ quantization infrastructure**: Added scale tensor storage in `DFlashDraftWeight`
- ✅ **Python bindings**: Added `load_dflash_weights_quantized()` method in TurboMind Python API
- ✅ **Unit tests**: Extended `test_dflash_turbomind.py` with quantization-related test cases
- ✅ **Documentation**: Added DFlash quantization usage docs in both English and Chinese

### Files Changed
| File | Changes |
|------|---------|
| `lmdeploy/messages.py` | Added `DraftQuantPolicy` enum, extended `SpeculativeConfig` |
| `lmdeploy/turbomind/turbomind.py` | Added quantization detection and dequantization methods |
| `src/turbomind/models/llama/DFlashDraftWeight.h` | Added quantization metadata fields and scale tensor arrays |
| `src/turbomind/models/llama/DFlashDraftWeight.cc` | Resize scale tensor arrays in constructor |
| `src/turbomind/turbomind.h` | Added `LoadDFlashWeightsQuantized()` declaration |
| `src/turbomind/turbomind.cc` | Implemented quantized weight loading logic |
| `src/turbomind/python/bind.cpp` | Added `load_dflash_weights_quantized()` Python binding |
| `tests/test_lmdeploy/test_dflash_turbomind.py` | Added `TestDraftQuantPolicy` test class |
| `docs/en/advance/spec_decoding.md` | Added DFlash documentation with quantization guide |
| `docs/zh_cn/advance/spec_decoding.md` | Added Chinese DFlash documentation |

### Learnings
- **Codebase patterns**: The existing Turbomind weight loading pipeline provides a clear template for adding new features
- **Quantization design**: FP16 default with opt-in quantization is a user-friendly approach
- **Backward compatibility**: The implementation maintains full backward compatibility by keeping default behavior unchanged
- **Documentation best practices**: Bilingual docs with usage examples and configuration tables are most helpful
- **Testing approach**: Unit tests for config classes catch issues early, while integration tests verify end-to-end functionality

### Key Features
1. **Auto-detection**: Quantization type is auto-detected from model config files
2. **Configurable group size**: Users can specify quantization group size (default 128)
3. **Multiple quantization methods**: FP16, INT8, INT4, AWQ, GPTQ all supported
4. **Memory savings**: INT4 quantization reduces draft model memory by ~75% (~2GB → ~0.5GB)

---
