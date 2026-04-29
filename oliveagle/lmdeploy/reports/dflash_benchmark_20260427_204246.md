# DFlash 性能基准测试报告

**生成时间**: 2026-04-27 20:42:46

## 测试配置

| 配置项 | 值 |
|--------|-----|
| **Target Model** | `/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B` |
| **Draft Model** | `/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash` |
| **Speculative Tokens** | 8 |
| **Output Tokens** | 256 |

## 系统信息

| 项目 | 值 |
|------|-----|
| PyTorch | 2.5.1+cu124 |
| CUDA | 12.4 |
| GPU | Tesla PG503-216 |
| GPU Memory | 31.7 GB |

## 性能对比

### Decode 速度 (tokens/s)

| Context | Baseline | DFlash | Speedup |
|---------|----------|--------|---------|
| 1K | 49.2 | 0.0 | 0.00x |
| 2K | 48.1 | 0.0 | 0.00x |
| 4K | 45.8 | 0.0 | 0.00x |
| 8K | 39.9 | 0.0 | 0.00x |

### 详细结果

| Context | Mode | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) |
|---------|------|--------------|-------------|---------------|---------------|----------|
| 1K | Baseline | 197.5 | 5208.0 | 3043.3 | 49.2 | 5.41 |
| 2K | Baseline | 251.8 | 5325.2 | 4770.2 | 48.1 | 5.58 |
| 4K | Baseline | 483.1 | 5587.6 | 5383.5 | 45.8 | 6.07 |
| 8K | Baseline | 1125.3 | 6415.8 | 4799.8 | 39.9 | 7.54 |

## 结论

