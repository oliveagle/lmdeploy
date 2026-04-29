# DFlash 性能基准测试报告

**生成时间**: 2026-04-27 23:33:07

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
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| GPU | Tesla PG503-216 |
| GPU Memory | 31.7 GB |

## 性能对比

### Decode 速度 (tokens/s)

| Context | Baseline | DFlash | Speedup |
|---------|----------|--------|---------|
| 1K | 41.4 | 0.0 | 0.00x |
| 2K | 32.3 | 0.0 | 0.00x |
| 4K | 17.6 | 0.0 | 0.00x |
| 8K | 6.5 | 0.0 | 0.00x |

### 详细结果

| Context | Mode | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) |
|---------|------|--------------|-------------|---------------|---------------|----------|
| 1K | Baseline | 753.1 | 6177.8 | 798.1 | 41.4 | 6.93 |
| 2K | Baseline | 2134.4 | 7935.8 | 562.7 | 32.3 | 10.07 |
| 4K | Baseline | 8118.2 | 14520.1 | 320.4 | 17.6 | 22.64 |
| 8K | Baseline | 31991.3 | 39533.1 | 168.8 | 6.5 | 71.52 |

## 结论

