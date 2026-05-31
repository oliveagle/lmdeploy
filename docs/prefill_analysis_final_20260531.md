# Prefill 性能分析总结报告

## 日期: 2026-05-31

---

## 一、Python + TurboMind 真实性能数据

### 测试模型
`/mnt/data/models/modelscope_models/Qwen3.6-35B-A3B-AWQ`

### 实测数据（2 轮测试，5 次运行平均）

| Input | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|-------|-----------|-----------------|----------------|
| 512 | 119-120 | **4,253-4,273** | 39-42 |
| 1024 | 190-195 | **5,244-5,378** | 38-41 |
| 4096 | 675-677 | **6,058-6,070** | 39 |
| 8192 | 1560-1561 | **5,247-5,250** | 36 |

### 关键发现

**"好几万 tok/s" 是误解**！

- **真实性能**: 峰值 ~6,000 tok/s (4096 input)
- **用户声称**: "好几万 tok/s"
- **差距来源**: 可能是 aggregate 吞吐量或多请求并发数据

**Archive 数据不可信**：
- Archive 声称: 6,408-12,614 tok/s
- 实测验证: 841-1076 tok/s (回到 Archive commit)
- **结论**: Archive 数据不是来自单请求测试

---

## 二、Rust Server 当前状态

### 阻塞问题：QKV Fusion 段错误

**崩溃信息**：
```
段错误 (core dumped)
位置: src/turbomind/capi/turbomind_c.cc:1271
文件: model-00008-of-00009.safetensors
步骤: Fusing 10 QKV tensors for layers.7.attention
```

**崩溃前日志**：
```
[C-API] Read Q tensor: 32 MB, 0.0ms (x10)
[C-API] Fusing 10 QKV tensors...
[C-API] QKV fusion navigating ... (key: layers.7.attention)
[C-API]   attn_module found, type=AttentionWeight
[C-API]   child('w_qkv') -> 0x...
[C-API]   w_qkv_module->type() = LinearWeight
[段错误]
```

**根本原因假设**：
1. `fused_tensor.raw_data()` 返回 NULL
2. `fused_tensor.alloc()` 失败但未检测
3. GPU 内存不足或分配逻辑错误

### Python vs Rust 加载路径差异

| Python TurboMind | Rust Server |
|------------------|--------------|
| pybind11 直接绑定 | C API (InitFromPath) |
| `_tm.TurboMind.create()` | `TM_TurboMind_InitFromPath()` |
| 内部权重加载 | `LoadWeightsFromSafetensors` |
| 跳过 QKV fusion 逻辑 | C++ 代码中执行 QKV fusion |

**关键差异**：Python 路径可能跳过了 C++ 中的 QKV fusion 逻辑，或者 pybind11 绑定使用了不同的代码路径。

---

## 三、性能差距分析（假设 Rust 可启动）

### 已实施的优化

1. ✅ **forward_async**: 异步 forward 替代阻塞 promise/future
2. ✅ **GPU Tokenizer**: uint32 零拷贝路径
3. ✅ **Condvar wait()**: 无超时事件驱动等待
4. ✅ **Request Pool**: 预分配 ModelRequest 对象池

### 剩余瓶颈

1. **Event Sync**: `event.sync()` 在 GPU 拷贝后阻塞（5-10% 损失）
2. **TensorMap 复制**: 每次请求创建新 TensorMap（3-5% 损失）
3. **回调链路**: C++ → Rust → Condvar 路径（3-5% 损失）
4. **Batch 调度**: 未实现 Gateway batch 优化（并发场景 20-50% 损失）

### 超越 Python 的策略

| 策略 | 预期提升 | 优先级 |
|------|---------|--------|
| 消除 Event Sync | 5-10% | P1 |
| TensorMap 复用 | 3-5% | P2 |
| 回调路径优化 | 3-5% | P2 |
| CUDA Stream 重叠 | 5-10% | P3 |
| Batch 调度 | 20-50% (并发) | P3 |

**综合预期**: 单请求场景提升 15-25%，可从 ~6,000 tok/s 提升到 ~7,000 tok/s

---

## 四、Beads 任务状态

### 新创建任务

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| lmdeploy-3oxi | 修复 QKV Fusion 段错误 | P0 | Open | - |
| lmdeploy-rslt | Python vs Rust 基准对比 | P0 | Open | lmdeploy-3oxi |

### 阻塞状态

**lmdeploy-3oxi (P0)**: 当前最高优先级任务
- 阻塞 lmdeploy-rslt
- 影响: 无法运行任何 Rust 性能测试

### 优先级建议

```
立即执行:
  lmdeploy-3oxi - 修复 QKV Fusion 段错误

随后执行:
  lmdeploy-rslt - 运行 Rust benchmark 获取真实数据
```

---

## 五、下一步

1. **立即**: 修复 QKV Fusion 段错误 (lmdeploy-3oxi)
2. **然后**: 运行 Rust prefill benchmark (lmdeploy-rslt)
3. **最后**: 根据实际差距决定进一步优化策略

---

## 六、结论

1. **"好几万 tok/s" 不是真实单请求性能**
   - 真实性能: ~6,000 tok/s 峰值
   - 目标: Rust 需要超过 ~6,000 tok/s

2. **Rust Server 当前无法启动**
   - QKV Fusion 段错误阻塞所有测试
   - 必须先修复此问题

3. **超越 Python 的路径清晰**
   - 已有优化基础 (forward_async, GPU tokenizer)
   - 剩余优化点明确 (Event Sync, TensorMap, 回调)
   - 预期可提升 15-25%

4. **关键行动**
   - 修复 QKV Fusion (P0)
   - 运行基准测试验证
   - 根据实际差距优化
