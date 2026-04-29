# DFlash 性能基准测试报告

**生成时间**: 2026-04-27 20:51:22

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
| 1K | 50.2 | 0.0 | 0.00x |
| 2K | 48.5 | 0.0 | 0.00x |
| 4K | 46.1 | 0.0 | 0.00x |
| 8K | 40.0 | 0.0 | 0.00x |

### 详细结果

| Context | Mode | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) |
|---------|------|--------------|-------------|---------------|---------------|----------|
| 1K | Baseline | 196.1 | 5098.3 | 3065.4 | 50.2 | 5.29 |
| 2K | Baseline | 285.6 | 5275.6 | 4205.4 | 48.5 | 5.56 |
| 4K | Baseline | 520.1 | 5555.7 | 5001.2 | 46.1 | 6.08 |
| 8K | Baseline | 1161.3 | 6397.8 | 4650.9 | 40.0 | 7.56 |

## 结论

