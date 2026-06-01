# Benchmark 测试状态总结 - Python + TurboMind vs Rust + TurboMind

## 任务目标

测试 Qwen3.6-35B-A3B-AWQ 模型在 4k, 8k, 16k context 下的 prefill 和 decode 性能，对比 Python + TurboMind 和 Rust + TurboMind 两种方式。

## 当前状态

### 问题 1: Python TurboMind API 破坏

**错误信息**:
```
TypeError: create(): incompatible function arguments. The following argument types are supported:
    1. (model_dir: str, engine_config: _turbomind.EngineConfig) -> _turbomind.TurboMind
```

**原因**: Python API (`lmdeploy/turbomind/turbomind.py`) 调用 `_tm.TurboMind.create()` 时使用的参数格式与 C++ pybind 绑定不匹配。

**影响**: 无法使用 Python TurboMind API 进行测试。

**修复方案**:
1. 更新 `turbomind.py` 第 653 行以匹配新的 API
2. 或者使用已转换的 TurboMind 模型绕过转换步骤

### 问题 2: Rust TurboMind MoE 权重加载失败

**错误信息**:
```
[C-API] DEBUG SKIP: tensor=model.language_model.layers.1.mlp.experts.0.down_proj.qweight -> tm_path=layers.1.moe_ffn.experts.0.w2.weight -> param=weight (slot not found)
```

**原因**: Qwen3.6-35B-A3B-AWQ 模型有 256 个专家，但 C++ 权重结构可能只创建了部分专家的槽位。

**影响**: Rust benchmark 无法加载模型权重。

**修复方案**:
1. 检查 `src/turbomind/models/llama/` 中的 MoE FFN 层权重结构
2. 确认 `expert_num` 配置与模型匹配
3. 可能需要更新权重映射逻辑

### 问题 3: Buffer data_ NULL 崩溃

**状态**: 已修复但未验证

**修复内容**:
- 提交 `17f12277`: 添加 NULL 检查到 `buffer.h:70`
- 提交 `8fabb2ad`: 修复 QKV Fusion 崩溃

**验证**: 需要使用可工作的 benchmark 验证修复是否有效。

## 已创建的文件

### Python Benchmark Scripts
- `/mnt/data/lmdeploy/benchmark/benchmark_python_turbomind_unified.py` - Python benchmark (4k, 8k, 16k)

### Rust Benchmark Binary
- `/mnt/data/lmdeploy/lmdeploy-rust-server/target/release/prefill_benchmark_4_8_16k` - Rust benchmark binary

### 测试配置
```python
NUM_REQUESTS = 5
WARMUP_REQUESTS = 2
OUTPUT_LENGTH = 512

TEST_SCENARIOS = [
    {"name": "4k", "input_len": 4096, "output_len": 512},
    {"name": "8k", "input_len": 8192, "output_len": 512},
    {"name": "16k", "input_len": 16384, "output_len": 512},
]
```

## 下一步行动

### 方案 A: 修复 Python API (推荐用于快速验证)
1. 修改 `lmdeploy/turbomind/turbomind.py:653` 使用正确的 API 格式
2. 运行 Python benchmark 获取基准数据
3. 修复 Rust MoE 权重加载问题
4. 运行 Rust benchmark
5. 生成对比报告

### 方案 B: 使用已转换模型 (如果存在)
1. 检查是否有已转换的 TurboMind 格式模型
2. 直接加载绕过转换步骤
3. 运行两个 benchmark

### 方案 C: 使用其他模型验证
1. 使用非 MoE 模型（如 Qwen2.5-7B）先验证 benchmark 脚本
2. 确认脚本正确后再解决 MoE 模型问题

## 环境信息

- **模型路径**: `/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ`
- **GPU**: Tesla PG503-216 (32GB)
- **CUDA**: 12.5
- **Python**: 3.13.9
- **LMDeploy**: 0.13.0 (development version)

## 相关文档

- `/mnt/data/lmdeploy/docs/rust_vs_python_prefill_performance_comparison_20260601.md` - 历史性能对比数据
- `/mnt/data/lmdeploy/docs/prefill_final_report_20260601.md` - Python prefill 基准数据

## 备注

由于代码库处于活跃开发状态，API 经常发生变化。建议：
1. 使用稳定版本的 LMDeploy 进行 benchmark
2. 或等待当前开发版本稳定后再测试
3. 或联系 LMDeploy 维护者确认正确的 API 用法
