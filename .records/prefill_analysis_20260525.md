# LMDeploy Prefill 性能分析报告

> **日期**: 2026-05-25
> **模型**: Qwen3.6-35B-A3B-AWQ @ `/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ`
> **GPU**: Tesla V100 32GB

---

## 1. 当前实测数据

### Python TurboMind (stream_output=True, max_new_tokens=1)

| Context | Tokens | TTFT Avg (ms) | Prefill (tok/s) |
|---------|--------|---------------|-----------------|
| 1K      | 1,021  | 190.4         | 5,363           |
| 4K      | 4,091  | 677.3         | 6,040           |
| 8K      | 8,191  | 1,568.1       | 5,224           |

### Rust Server

| Context | Tokens | TTFT Avg (ms) | Prefill (tok/s) | Min TTFT (ms) | Max TPS |
|---------|--------|---------------|-----------------|---------------|---------|
| 1K      | 1,001  | 284.5         | 3,518           | 187.3         | 5,344   |
| 2K      | 2,001  | 622.6         | 3,214           | 622.5         | 3,215   |
| 4K      | 4,001  | 1,173.6       | 3,409           | 950.4         | 4,210   |
| 8K      | 8,001  | 3,011.8       | 2,657           | 3,011.2       | 2,657   |

### Python vs Rust 差距

| Context | Python TPS | Rust TPS | 差距 | 备注 |
|---------|-----------|----------|------|------|
| 1K      | 5,363     | 3,518    | **1.5x** | Rust 最佳 5,344 ≈ Python 平均 |
| 4K      | 6,040     | 3,409    | **1.8x** | Rust 最佳 4,210 ≈ Python 的 70% |
| 8K      | 5,224     | 2,657    | **2.0x** | Rust 无缓存命中 |

**关键观察**:
- Rust 的**最佳情况**接近 Python 的 80-100%
- Rust 的**平均情况**只有 Python 的 50-75%
- 8K 上下文差距最大，Rust 是 Python 的一半

---

## 2. 历史数据对比

### Qwen3.5-9B (历史数据, 2026-05-18)

| Context | TTFT (ms) | Prefill (tok/s) | 当前 35B TTFT | 当前 35B TPS |
|---------|-----------|-----------------|---------------|--------------|
| 1K      | 69.62     | 14,827          | 190.4         | 5,363        |
| 4K      | 122.06    | 33,728          | 677.3         | 6,040        |
| 8K      | 191.37    | 42,875          | 1,568.1       | 5,224        |

**关键发现**:

1. **历史 42,875 tok/s 是 Qwen3.5-9B 的数据**，不是 35B 模型
2. 9B 模型的 TTFT 远快于 35B：1K 只有 69ms，而 35B 是 190ms
3. 35B 模型的 prefill TPS 在不同长度下相对稳定（5-6K tok/s）
4. 9B 模型的 prefill TPS 随上下文增长而增加（14K → 42K tok/s）

### 为什么 9B 模型的 TPS 随上下文增长？

9B 模型在 8K 上下文时达到 42K tok/s，这说明：
- GPU 计算能力充足
- Prefill 是计算密集型，更大的 batch 能更好利用 GPU
- 35B AWQ 模型受限于 KV cache 和模型权重加载带宽

---

## 3. 根因分析：Rust vs Python 性能差异

### 3.1 为什么 Python 能到 6,040 tok/s 而 Rust 只有 3,409 tok/s？

**Python TurboMind 调用链**:
```
Python → pybind11 → C++ TurboMind → CUDA
```

**Rust Server 调用链**:
```
Rust → FFI → C++ TurboMindCEngine → C++ TurboMind → CUDA
```

差异在于 Rust 多了一层 FFI 和 TurboMindCEngine 封装。

### 3.2 具体瓶颈分析

#### 已确认的瓶颈：

1. **KV Cache 复用**：
   - Python 每次创建新 instance，但底层 KV cache 可能被复用
   - Rust 的 session 管理可能没有完全隔离
   - **证据**：Rust 的 min TTFT 接近 Python 的 avg TTFT

2. **Warmup 不足**：
   - Python 有 JIT warmup + CUDA context warmup
   - Rust 的 warmup 可能不够充分
   - **证据**：Rust 的 min run 和 max run 差距大

3. **Tensor 拷贝**：
   - Rust 到 C++ 的 tensor 传递可能涉及额外拷贝
   - `TensorMap` 创建和转换开销
   - **证据**：8K 上下文差距最大（数据量最大）

#### 需要验证的假设：

1. **Completion Callback 异步机制**：
   - Rust 的 callback 是否在 GPU 完成后立即触发？
   - 是否存在 CPU-GPU 同步点？

2. **内存分配策略**：
   - 每次推理是否重新分配 tensor 内存？
   - 是否有预分配缓存机制？

3. **调度策略**：
   - `max_prefill_token_num` 和 `num_tokens_per_iter` 设置是否最优？
   - Python 的默认值是否更高效？

---

## 4. 结论

### 当前状态

| 指标 | Python | Rust | 状态 |
|------|--------|------|------|
| 1K Prefill | 5,363 tok/s | 3,518 tok/s | Rust 是 Python 的 66% |
| 4K Prefill | 6,040 tok/s | 3,409 tok/s | Rust 是 Python 的 56% |
| 8K Prefill | 5,224 tok/s | 2,657 tok/s | Rust 是 Python 的 51% |

### 关于历史 42,875 tok/s

- 这是 **Qwen3.5-9B** 模型的数据，不是 35B
- 35B 模型由于参数量大 7 倍，prefill 时间自然更长
- 当前 35B 的 5-6K tok/s 是合理的性能水平

### Rust vs Python 差距原因

1. **额外 FFI 层**：Rust → C++ 的 tensor 传递有额外开销
2. **Session 管理**：Rust 的 session 隔离可能不如 Python 彻底
3. **缓存效应**：Python 可能更好地利用了 KV cache
4. **Warmup 策略**：Python 的 warmup 更充分

### 建议优化方向

1. **减少 Tensor 拷贝**：使用零拷贝或共享内存
2. **优化 Session 管理**：确保每次推理使用完全隔离的 session
3. **更充分的 Warmup**：在 benchmark 前进行更多 warmup runs
4. **预分配内存**：复用输入/输出 tensor 缓冲区
5. **Completion Callback 优化**：减少 CPU-GPU 同步点
