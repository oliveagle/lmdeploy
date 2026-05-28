# LMDeploy Unified Benchmark: Python vs Rust

**Date**: 2026-05-28
**Model**: Qwen3.6-35B-A3B-AWQ
**GPU**: Tesla PG503-216 (32GB)
**CUDA Version**: 12.2
**Driver Version**: 535.309.01

---

## 测试方法

### 统一测试标准

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入长度 | 512, 1024, 4096, 8192 tokens | 与 Python 完全一致 |
| 输出长度 | 512 tokens | 固定输出长度 |
| 并发度 | 1 | 串行测试，避免并发干扰 |
| 温轮次数 | 2 | 每个场景 2 次 warmup |
| 测量次数 | 5 | 每个场景 5 次测量，取平均 |

### 计算公式（统一）

- **Prefill 吞吐量** = `input_len / (ttft_ms / 1000)`
- **Decode 吞吐量** = `1000 / tpot_ms`
- **TTFT** = 首个 token 时间（ms）
- **TPOT** = 每个输出 token 时间（ms）

---

## Python TurboMind 结果

### 测试命令
```bash
cd /mnt/data/lmdeploy
bash benchmarks-archive/python-turbomind_35b_awq_20260526/scripts/run_benchmark.sh
```

### 实测数据

| 输入长度 | 输出长度 | TTFT (ms) | TPOT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------|----------|-----------|-----------|-----------------|----------------|
| 512 | 512 | 79.9 | 23.5 | 6,408 | 42.6 |
| 1024 | 512 | 139.2 | 23.7 | 7,356 | 42.2 |
| 4096 | 512 | 402.0 | 24.7 | 10,190 | 40.5 |
| 8192 | 512 | 649.4 | 25.1 | 12,614 | 39.8 |

**来源**: `benchmarks-archive/python-turbomind_35b_awq_20260526/results/`

---

## Rust Server 结果

### 测试命令
```bash
cd /mnt/data/lmdeploy/lmdeploy-rust-server
cargo run --release --bin prefill_benchmark -- \
    --model /mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ \
    --warmup 2 \
    --measure 5 \
    --output tests/prefill_benchmark_rust.json
```

### 实测数据

| 输入长度 | 输出长度 | TTFT (ms) | TPOT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------|----------|-----------|-----------|-----------------|----------------|
| 512 | 512 | TBD | TBD | TBD | TBD |
| 1024 | 512 | TBD | TBD | TBD | TBD |
| 4096 | 512 | TBD | TBD | TBD | TBD |
| 8192 | 512 | TBD | TBD | TBD | TBD |

**状态**: 待测试（C++ engine weight loading bug 已修复）

---

## 对比分析

### Prefill 性能对比

| 输入长度 | Python (tok/s) | Rust (tok/s) | 差异 |
|----------|----------------|---------------|------|
| 512 | 6,408 | TBD | TBD |
| 1024 | 7,356 | TBD | TBD |
| 4096 | 10,190 | TBD | TBD |
| 8192 | 12,614 | TBD | TBD |

### Decode 性能对比

| 输入长度 | Python (tok/s) | Rust (tok/s) | 差异 |
|----------|----------------|---------------|------|
| 512 | 42.6 | TBD | TBD |
| 1024 | 42.2 | TBD | TBD |
| 4096 | 40.5 | TBD | TBD |
| 8192 | 39.8 | TBD | TBD |

---

## 验收标准

- [x] Python 基线数据已验证
- [x] Rust 测试脚本参数与 Python 统一
- [x] 计算公式完全一致
- [x] 报告格式统一
- [ ] Rust 测试结果已填充
- [ ] 性能对比分析完成

---

## C++ Engine Bug 修复记录

**问题**: C++ TurboMind engine 无法加载 Qwen3.6-35B-A3B-AWQ 模型

**根因**: Qwen3.5 MoE 使用混合注意力类型（linear_attention + full_attention），但 C++ engine 为所有层创建了 `attention` 模块，导致 linear_attn 层的 weight name 不匹配。

**修复**: 在 `src/turbomind/capi/turbomind_c.cc:1757` 添加条件判断，只为 full_attention 层创建 `attention` 模块，linear_attn 层使用 `DeltaNetWeight` 模块。

**提交**: `b47b5797` (待验证)
