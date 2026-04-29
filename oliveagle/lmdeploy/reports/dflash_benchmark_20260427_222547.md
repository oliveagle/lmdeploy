# DFlash 性能基准测试报告

**生成时间**: 2026-04-27 22:25:47

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
| 1K | 49.7 | 0.0 | 0.00x |
| 2K | 48.3 | 0.0 | 0.00x |
| 4K | 44.6 | 0.0 | 0.00x |
| 8K | 40.2 | 0.0 | 0.00x |

### 详细结果

| Context | Mode | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) |
|---------|------|--------------|-------------|---------------|---------------|----------|
| 1K | Baseline | 198.6 | 5152.0 | 3025.8 | 49.7 | 5.35 |
| 2K | Baseline | 287.3 | 5300.4 | 4179.6 | 48.3 | 5.59 |
| 4K | Baseline | 527.3 | 5741.2 | 4932.6 | 44.6 | 6.27 |
| 8K | Baseline | 1136.8 | 6373.9 | 4751.2 | 40.2 | 7.51 |

## 结论

