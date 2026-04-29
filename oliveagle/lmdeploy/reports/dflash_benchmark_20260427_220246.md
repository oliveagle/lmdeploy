# DFlash 性能基准测试报告

**生成时间**: 2026-04-27 22:02:46

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
| 1K | 50.3 | 0.0 | 0.00x |
| 2K | 48.9 | 0.0 | 0.00x |
| 4K | 46.0 | 0.0 | 0.00x |
| 8K | 40.0 | 0.0 | 0.00x |

### 详细结果

| Context | Mode | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) |
|---------|------|--------------|-------------|---------------|---------------|----------|
| 1K | Baseline | 185.3 | 5092.2 | 3244.2 | 50.3 | 5.28 |
| 2K | Baseline | 284.9 | 5238.0 | 4215.7 | 48.9 | 5.52 |
| 4K | Baseline | 555.7 | 5562.5 | 4680.6 | 46.0 | 6.12 |
| 8K | Baseline | 1160.5 | 6392.8 | 4653.9 | 40.0 | 7.55 |

## 结论

