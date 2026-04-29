# DFlash 性能基准测试报告

**生成时间**: 2026-04-27 23:00:00

## 测试配置

| 配置项 | 值 |
|--------|-----|
| **Target Model** | Qwen3.5-9B |
| **Draft Model** | Qwen3.5-9B-DFlash |
| **Speculative Tokens** | 8 |
| **Output Tokens** | 256 |

## 系统信息

| 项目 | 值 |
|------|-----|
| **GPU** | Tesla PG503-216 (V100) |
| **GPU Memory** | 31.7 GB |
| **CUDA Capability** | 7.0 (SM_70) |
| **PyTorch** | 2.5.1+cu124 |
| **CUDA** | 12.4 |
| **Triton** | 3.1.0 |

## 关键发现

### Turbomind + DFlash = 不可用

**Turbomind 不支持 speculative decoding**。

lmdeploy 的 DFlash 实现完全依赖 PyTorch backend (`PytorchEngineConfig`)，而 PyTorch backend 的 `flash_attn_varlen_func` 需要 **SM_80+ (A100/H100)**。

| Backend | 性能 | DFlash 支持 |
|---------|------|------------|
| Turbomind | 46.1 t/s (decode) | ❌ 不支持 |
| PyTorch (SM_80+) | ~46 t/s | ✅ 可用 |
| PyTorch (SM_70 / V100) | ~46 t/s | ❌ Flash Attention 不兼容 |

## 性能对比

### Decode 速度 (tokens/s)

| Context | Turbomind | DFlash (PyTorch) | Speedup |
|---------|-----------|------------------|---------|
| 1K | 49.8 | N/A | - |
| 2K | 48.5 | N/A | - |
| 4K | 46.0 | N/A | - |
| 8K | 40.1 | N/A | - |

### Prefill 速度 (tokens/s)

| Context | Turbomind |
|---------|-----------|
| 1K | 3,191 |
| 2K | 4,207 |
| 4K | 5,001 |
| 8K | 4,767 |

### 详细结果

| Context | Mode | Prefill (ms) | Prefill (t/s) | Decode (ms) | Decode (t/s) | Total (s) |
|---------|------|--------------|---------------|-------------|---------------|----------|
| 1K | Turbomind | 188.3 | 3,191 | 5,144.8 | 49.8 | 5.33 |
| 2K | Turbomind | 285.5 | 4,207 | 5,280.8 | 48.5 | 5.57 |
| 4K | Turbomind | 520.1 | 5,001 | 5,566.0 | 46.0 | 6.09 |
| 8K | Turbomind | 1,133.0 | 4,767 | 6,378.6 | 40.1 | 7.51 |

## 结论

- **Turbomind**: 46.1 tokens/s 平均 decode 速度，不支持 DFlash
- **DFlash**: 需要 PyTorch backend + SM_80+ GPU (A100+)
- **V100 (SM_70)**: 不支持 Flash Attention，DFlash 无法运行

## 测试脚本

- **文件**: `/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy/benchmark_dflash_standard.py`
- **特性**: 子进程隔离、Turbomind/PyTorch 双 backend、Markdown + JSON 报告
