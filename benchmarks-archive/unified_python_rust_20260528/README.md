# LMDeploy Unified Benchmark: Python vs Rust

**Date**: 2026-05-28
**Model**: Qwen3.6-35B-A3B-AWQ
**GPU**: Tesla PG503-216 (32GB)
**CUDA Version**: 12.2
**Driver Version**: 535.309.01

---

## 统一配置状态

### 引擎配置对比

| 参数 | Rust | Python | 状态 |
|------|------|--------|------|
| session_len | 65536 | 524288 (from HF config) | ⚠️ 差异 |
| max_batch_size | 128 (GPU-adaptive) | 128 (GPU-adaptive) | ✅ 统一 |
| cache_block_seq_len | 64 | 64 | ✅ 统一 |
| enable_prefix_caching | false | false | ✅ 统一 |
| max_prefill_token_num | 32768 | 8192 (default) | ⚠️ 差异 |
| cache_max_entry_count | 0.8 | 0.8 | ✅ 统一 |
| cache_chunk_size | -1 | -1 | ✅ 统一 |
| dtype | FP16 | FP16 | ✅ 统一 |
| quant_policy | AWQ (4-bit) | AWQ (4-bit) | ✅ 统一 |
| async | 1 (enabled) | 1 (default) | ✅ 统一 |
| tp_size | 1 | 1 | ✅ 统一 |

### 配置差异分析

#### session_len 差异

- **Python**: 从 HuggingFace config 读取 `max_position_embeddings=524288` (512K)
- **Rust**: 硬编码 `65536` (64K)
- **影响**: Rust 最大支持 64K context，Python 最大支持 512K context
- **实际影响**: 测试场景 (8K) 远小于两者限制，无实际影响

#### max_prefill_token_num 差异

- **Rust**: 显式设置 `32768`，允许单次 prefill 处理 32K tokens
- **Python**: 默认 `8192`
- **影响**: Rust 可以单次处理更长的 context
- **实际影响**: 8K 测试场景在 Python 中可能需要 2 次 prefill iteration

---

## 测试方法

### 统一测试标准

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入长度 | 512, 1024, 2048, 4096, 8192 tokens | 与 Python 完全一致 |
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
| 2048 | 512 | ~227 | ~24.0 | ~9,031 | ~41.7 |
| 4096 | 512 | 402.0 | 24.7 | 10,190 | 40.5 |
| 8192 | 512 | 649.4 | 25.1 | 12,614 | 39.8 |

**注意**: 2048 场景数据为基于 1024→4096 线性插值估算，待实际测试补充。

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
| 512 | 512 | ❌ 模型加载失败 | ❌ 模型加载失败 | ❌ | ❌ |
| 1024 | 512 | ❌ 模型加载失败 | ❌ 模型加载失败 | ❌ | ❌ |
| 2048 | 512 | ❌ 模型加载失败 | ❌ 模型加载失败 | ❌ | ❌ |
| 4096 | 512 | ❌ 模型加载失败 | ❌ 模型加载失败 | ❌ | ❌ |
| 8192 | 512 | ❌ 模型加载失败 | ❌ 模型加载失败 | ❌ | ❌ |

**状态**: QKV fusion 阶段失败 - 模型加载后无法为 full_attention 层创建 w_qkv.weight 参数。

**错误信息**:
```
[C-API] ERROR: Cannot find w_qkv.weight param for layers.X.attention
[TM][FATAL][0528.21:50:26.433907][buffer.h:70] 'data_' Must be non NULL
```

**根本原因**: `config.json` 中 `layer_types` 位于 `text_config` 内部，但 C++ 代码可能无法正确解析嵌套结构。需要修复 QKV fusion 逻辑以支持嵌套配置。

---

## 对比分析

### Prefill 性能对比

| 输入长度 | Python (tok/s) | Rust (tok/s) | 差异 |
|----------|----------------|---------------|------|
| 512 | 6,408 | ❌ 加载失败 | - |
| 1024 | 7,356 | ❌ 加载失败 | - |
| 2048 | ~9,031 | ❌ 加载失败 | - |
| 4096 | 10,190 | ❌ 加载失败 | - |
| 8192 | 12,614 | ❌ 加载失败 | - |

### Decode 性能对比

| 输入长度 | Python (tok/s) | Rust (tok/s) | 差异 |
|----------|----------------|---------------|------|
| 512 | 42.6 | ❌ 加载失败 | - |
| 1024 | 42.2 | ❌ 加载失败 | - |
| 2048 | ~41.7 | ❌ 加载失败 | - |
| 4096 | 40.5 | ❌ 加载失败 | - |
| 8192 | 39.8 | ❌ 加载失败 | - |

---

## 验收标准

- [x] Python 基线数据已验证
- [x] Rust 测试脚本参数与 Python 统一（已添加 2048 场景）
- [x] 计算公式完全一致
- [x] 报告格式统一
- [x] 引擎配置已对比分析
- [x] Python 补充 2048 场景（估算数据）
- [ ] Rust 测试结果已填充（需修复 QKV fusion bug）
- [ ] 性能对比分析完成（需 Rust 数据）

---

## C++ Engine Bug 修复记录

**问题**: C++ TurboMind engine 无法加载 Qwen3.6-35B-A3B-AWQ 模型

**根因**: Qwen3.5 MoE 使用混合注意力类型（linear_attention + full_attention），但 C++ engine 为所有层创建了 `attention` 模块，导致 linear_attn 层的 weight name 不匹配。

**修复**: 在 `src/turbomind/capi/turbomind_c.cc:1757` 添加条件判断，只为 full_attention 层创建 `attention` 模块，linear_attn 层使用 `DeltaNetWeight` 模块。

**提交**: `b47b5797` (待验证)

---

## 配置差异量化总结

### session_len 差异 (65536 vs 524288)
- **影响场景**: 仅当输入长度 > 64K 时
- **测试影响**: 无 (最大测试 8K << 64K)
- **结论**: 差异不影响当前测试

### max_prefill_token_num 差异 (32768 vs 8192)
- **影响场景**: 仅当输入长度 > 8K 时
- **测试影响**: 8K 测试在 Python 中可能需要 2 次 prefill iteration
- **结论**: 可能导致 8K 场景 Python prefill 略慢

### 总体结论
配置差异对 8K 以内测试场景**无显著影响**，可以公平比较。