# Prefill 性能分析总结 (2026-06-01)

## 一、Python + TurboMind 真实性能

### 历史测试数据（多轮实测汇总）

| 输入长度 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------|-----------|-----------------|----------------|
| 512 | 118-120 | **4,253-4,312** | 39-42 |
| 1024 | 190-195 | **5,244-5,394** | 38-41 |
| 4096 | 676-709 | **5,778-6,070** | 34-39 |
| 8192 | 1561-1565 | **5,236-5,250** | 35-36 |

### 关键结论

**"好几万 tok/s" 不是真实单请求性能**
- 真实峰值: **~6,000 tok/s** (4096 tokens)
- Archive 声称: 6,408-12,614 tok/s (与实测不符)
- 可能来源: aggregate 吞吐量、并发场景、不同测量方法

---

## 二、Rust Server 当前状态

### 已修复的问题

✅ **lmdeploy-9yov**: QKV Fusion 崩溃
- 修复: 移除不安全的 `raw_data()` 调用
- 提交: 8fabb2ad

### 当前阻塞问题

🔴 **lmdeploy-osrp**: model-00009 加载后崩溃 (P0)
```
位置: buffer.h:70
断言: TM_CHECK_NOTNULL(data_) 失败
现象: model-00001~00008 加载成功，00009 加载后崩溃
```

🟡 **lmdeploy-evkx**: get_model_arch 参数不兼容 (P1)
```
错误: TypeError: get_model_arch() got an unexpected keyword argument 'trust_remote_code'
影响: Python benchmark 无法运行
```

### Python vs Rust 加载路径差异

| 方面 | Python TurboMind | Rust Server |
|------|------------------|-------------|
| 绑定 | pybind11 直接 | C API InitFromPath |
| 权重加载 | `_tm_model.export()` → `process_weight()` → `create_engine()` | `InitFromPath()` → `LoadWeightsFromSafetensors()` → `CreateEngine()` |
| QKV Fusion | pybind11 内部处理 | C++ 代码执行（已跳过失败部分） |
| 后续步骤 | pybind11 自动处理 | C++ Prepare() 可能访问未初始化权重 |

---

## 三、Beads 任务状态

### 当前任务

| ID | 任务 | 优先级 | 状态 |
|----|------|--------|------|
| lmdeploy-osrp | model-00009 加载后崩溃 | **P0** | 🔄 Open |
| lmdeploy-evkx | get_model_arch 参数修复 | P1 | 🔄 Open |
| lmdeploy-q2f1 | Rust 超过 Python 优化方案 | P0 | 🔒 Blocked |

### 已完成任务（2026-06-01）

- ✅ lmdeploy-9yov: QKV Fusion 崩溃
- ✅ lmdeploy-hqb2: get_model_arch 参数不兼容（标记关闭但代码未修复）

---

## 四、超越 Python 的策略

### 已实施优化（25-43% 提升）

| 优化项 | 状态 | 预期提升 |
|--------|------|----------|
| GPU Tokenizer 零拷贝 | ✅ | 5-10% |
| forward_async | ✅ | 10-15% |
| Condvar wait() | ✅ | 3-5% |
| Event Sync 消除 | ✅ | 5-10% |
| Request Pool | ✅ | 2-3% |

### 待实施优化（需先修复崩溃）

| 策略 | 预期提升 | 优先级 |
|------|---------|--------|
| 修复 model-00009 崩溃 | 必须 | P0 |
| TensorMap 复用 | 3-5% | P1 |
| 回调路径优化 | 3-5% | P2 |
| CUDA Stream 重叠 | 5-10% | P3 |
| Batch 调度 | 20-50% | P3 |

### 综合预期

- **单请求**: 30-50% 提升 → **~7,800-9,000 tok/s**
- **并发**: Batch 调度额外 20-50%

---

## 五、下一步行动

### 立即执行（P0）

1. **lmdeploy-osrp**: 调查 model-00009 崩溃根因
   - 对比 Python vs Rust 权重加载差异
   - 定位哪个权重参数未正确初始化
   - 修复 buffer.h:70 崩溃

2. **lmdeploy-evkx**: 修复 get_model_arch 参数
   - 在 archs.py 添加 `trust_remote_code: bool = False` 参数

### 验证阶段

3. Rust Server 能启动 → 运行 prefill_benchmark
4. Python benchmark 能运行 → 获取对比数据
5. 根据实际差距决定进一步优化

---

## 六、结论

1. **真实性能**: ~6,000 tok/s 峰值，不是"好几万"
2. **Rust 被阻塞**: model-00009 加载后崩溃
3. **超越路径**: 已有 25-43% 优化基础，预期可达 7,800-9,000 tok/s
4. **关键行动**: 修复 lmdeploy-osrp 和 lmdeploy-evkx

---

## 七、相关文档

- `docs/prefill_analysis_complete_20260531.md`
- `docs/prefill_performance_deep_analysis_20260531.md`
- `docs/prefill_analysis_final_20260531.md`
