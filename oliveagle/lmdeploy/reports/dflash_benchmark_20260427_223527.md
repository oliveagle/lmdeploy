# DFlash 性能基准测试报告

**生成时间**: 2026-04-27 22:35:27

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
| 1K | 49.8 | 0.0 | 0.00x |
| 2K | 48.5 | 0.0 | 0.00x |
| 4K | 46.0 | 0.0 | 0.00x |
| 8K | 40.1 | 0.0 | 0.00x |

### 详细结果

| Context | Mode | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) |
|---------|------|--------------|-------------|---------------|---------------|----------|
| 1K | Baseline | 188.3 | 5144.8 | 3191.0 | 49.8 | 5.33 |
| 2K | Baseline | 285.5 | 5280.8 | 4206.7 | 48.5 | 5.57 |
| 4K | Baseline | 520.1 | 5566.0 | 5001.2 | 46.0 | 6.09 |
| 8K | Baseline | 1133.0 | 6378.6 | 4767.1 | 40.1 | 7.51 |

## 结论

