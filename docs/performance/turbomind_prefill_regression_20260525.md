# TurboMind 8K Prefill 性能退化分析报告

**分析日期**: 2026-05-25
**基准日期**: 2026-05-18
**基准**: `BENCHMARK_PYTHON_TM_20260518.json`

---

## 1. 退化数据对比

| 指标 | 历史基准 (2026-05-18) | 当前 (2026-05-24) | 退化倍数 |
|------|----------------------|-------------------|---------|
| 4K prefill | 33,727 tok/s (122ms TTFT) | 3,716 tok/s (1,076ms) | ~9x |
| 8K prefill | 42,875 tok/s (191ms TTFT) | 2,666 tok/s (3,001ms) | ~16x |

测试模型: `Qwen3.6-35B-A3B-AWQ`，GPU: `Tesla V100 32GB`

### 当前 benchmark 结果 (2026-05-24)

| 时间 | Benchmark 文件 | 8K prefill | 说明 |
|------|---------------|-----------|------|
| 22:48 | `prefill_benchmark_awq_35b_final.json` | 2,666 tok/s | **退化的当前值** |
| 23:11 | `prefill_benchmark_final_v3.json` | 2,666 tok/s | 复现确认 |
| 23:20 | `benchmark_python_official.json` | TTFT=0 (bug) | 脚本 bug，无有效数据 |

---

## 2. 自基准以来提交摘要

自 2026-05-18 基准日期以来有 50 个提交，按相关性排序的关键提交:

### 2.1 高怀疑度提交

| Commit | 描述 | 文件 | 相关性 |
|--------|------|------|--------|
| `8c1c0db7` | KV Cache 直接映射 - 消除 tmp_kv 拷贝 | `unified_attention_layer.cc` | ⭐⭐⭐ 核心 |
| `037bdf5a` | 优化 attention kernel 内存效率 | `unified_attention_layer.cc` | ⭐⭐⭐ 核心 |
| `10ee4f1b` | Stream 优化 - 减少 CUDA 同步点 | 多处 | ⭐⭐ 可能 |
| `d561a07c` | 修复 GatedDeltaNetLayer OOM | `GatedDeltaNetLayer.cc` | ⭐ 低 |

### 2.2 低怀疑度提交

| Commit | 描述 | 影响范围 |
|--------|------|---------|
| `e59eb5de` | 修复 32K context OOM | session_len 配置，可能间接影响 |
| `e110b13b` | Rust+C++ 性能根因分析 | 仅 lmdeploy-rust，不影响 Python |
| `cf678b2a` | Python vs Rust 吞吐量差距 | 仅分析文档 |

---

## 3. 根因分析

### 3.1 主要嫌疑人: `8c1c0db7` - KV Cache 直接映射

**变更内容**: 删除了 `tmp_kv` 临时 tensor 分配 + `invokeFlattenKV_v2_` 调用，改为直接从 QKV tensor 的 offset 读取 K/V 数据。

**原始代码**（删除前）:
```cpp
// 分配临时 KV 并展平为 kernel 友好格式
Tensor tmp_kv{{local_kv_head_num, is_mla ? 1 : 2, d.prefill.k_sum + MAX_CTA_S, size_per_head}, dtype, device};
// ...
invokeFlattenKV_v2_(params, d.prefill.k_sum);  // 展平 K/V
```

**新代码**:
```cpp
const char* k_data = (const char*)qkv.raw_data() + local_head_num * size_per_head * byte_size(dtype, 1);
// 直接读取 QKV offset，跳过展平
// invokeFlattenKV_v2_ 被注释掉
```

**可能问题**:
1. **内存对齐**: 从 QKV tensor 的 offset 直接读取可能未满足 CUDA kernel 的对齐要求
2. **layout 不匹配**: QKV tensor 的 layout 可能不是 attention kernel 期望的 contiguous 格式
3. **缺少展平操作**: `invokeFlattenKV_v2_` 可能不仅是优化，还负责 layout 转换

### 3.2 次要嫌疑人: `037bdf5a` - Attention Kernel 内存优化

该提交在 `8c1c0db7` 之上进一步修改 attention kernel，降低 O(N²) 内存压力。可能引入额外同步或改变 kernel launch 配置。

### 3.3 其他可能原因

- **Python pipeline 脚本 bug**: `benchmark_python_official.py` 中 TTFT 捕获逻辑有缺陷（`generate_token_len` 初始为 0 导致不触发），但这不影响 `prefill_benchmark_awq.py` 的直接 benchmark
- **配置变更**: `session_len`、`cache_max_entry_count` 等参数变化可能导致 cache 行为改变

---

## 4. 验证建议

### 4.1 优先级排序

1. **复现基准**: checkout 到 2026-05-18 的提交（`88ef247e` 附近），运行 `prefill_benchmark_awq.py`，确认 42K tok/s 可复现
2. **回退验证**: checkout 到 `8c1c0db7`（KV Cache 优化），运行同一 benchmark，确认退化出现
3. **当前 HEAD 验证**: 确认当前性能数据

### 4.2 具体命令

```bash
# 1. 基准复现
git checkout 88ef247e
cd lmdeploy-rust-server/tests
python prefill_benchmark_awq.py

# 2. 回退到 KV Cache 优化提交
git checkout 8c1c0db7
python prefill_benchmark_awq.py

# 3. 当前 HEAD
git checkout ralph-session/17794723
python prefill_benchmark_awq.py
```

### 4.3 深入验证

- **nvprof/ncu 分析**: 对 `88ef247e` 和 `HEAD` 分别跑 ncu profiler，比较 attention kernel 的 SM occupancy、memory bandwidth
- **CUDA event timing**: 在 `unified_attention_layer.cc` 的 `dispatchAttention` 前后加 CUDA event 计时
- **Layout 验证**: 检查 QKV tensor layout 是否与 attention kernel 期望一致

---

## 5. 修复方向

### 5.1 快速修复（如果确认是 8c1c0db7 导致）

- 回退 `8c1c0db7` 的 attention layer 修改，恢复 `tmp_kv` + `invokeFlattenKV_v2_`
- 验证性能是否恢复到 ~42K tok/s
- 后续用其他方式优化（如 pinned memory + async copy）

### 5.2 长期修复

- 分析 attention kernel 的 memory access pattern
- 实现零拷贝的 contiguous layout（如果 QKV 本身是 contiguous 的）
- 添加 `--enable-direct-kv` 编译选项，允许在 layout 允许时跳过展平

---

## 6. 附加发现

### 6.1 Benchmark 脚本问题

`benchmark_python_official.py` 的 TTFT 捕获逻辑有误：
```python
if ttft_time is None and response.generate_token_len:
    # generate_token_len 初始为 0，条件永远不满足
    ttft_time = time.perf_counter() - start_time
```

导致 4K/8K 的 `ttft_ms_avg = 0.0`。应改为：
```python
if ttft_time is None and response.status in (1, 2):
    ttft_time = time.perf_counter() - start_time
```

`prefill_benchmark_awq.py` 没有这个问题（使用 `out.status.value in (1, 2)` 判断）。

### 6.2 测试环境差异

- 历史基准使用 `Qwen3.6-35B-A3B-AWQ`
- 对比测试也使用同一模型
- 但 GPU 型号可能有变化（历史基准是 V100 32GB PG503-216，需确认当前 GPU）
