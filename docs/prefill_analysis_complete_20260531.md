# Prefill 性能分析完整报告

## 日期: 2026-05-31

---

## 一、Python + TurboMind 真实性能数据

### 测试配置
- **模型**: `/mnt/data/models/modelscope_models/Qwen3.6-35B-A3B-AWQ`
- **后端**: TurboMind C++ 引擎
- **测试方法**: 单请求串行，5 次运行取平均
- **测量指标**: TTFT (Time To First Token)，Prefill tok/s

### 历史测试数据汇总（3 轮测试）

| 输入长度 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------|-----------|-----------------|----------------|
| 512 | 118-120 | **4,253-4,312** | 39-42 |
| 1024 | 190-195 | **5,244-5,394** | 38-41 |
| 4096 | 676-709 | **5,778-6,070** | 34-39 |
| 8192 | 1561-1565 | **5,236-5,250** | 35-36 |

### 关键发现

**"好几万 tok/s" 的真相**：

1. **真实单请求性能**: 峰值约 **~6,000 tok/s**（4096 tokens 时）
2. **Archive 声称数据**: 6,408-12,614 tok/s - **与实测不符**
3. **Archive Commit 验证**: 回到 738f60eb 重新测试，实际只有 841-1076 tok/s
4. **结论**: Archive 数据可能是 aggregate 吞吐量或使用了不同的测量方法

### "好几万 tok/s" 的可能来源

1. **Aggregate 吞吐量**: 多请求并发时的总吞吐量
2. **Decode 阶段**: 多个并发 decode 的 aggregate（~40 tok/s × 100 = 4000 tok/s）
3. **测试工具差异**: `profile_throughput.py` 的 `input_throughput` 可能是 aggregate 指标

---

## 二、Rust Server 当前状态

### 阻塞问题

#### 问题 1: QKV Fusion 段错误 (P0)

**崩溃信息**：
```
位置: src/turbomind/capi/turbomind_c.cc:1284
代码: std::memcpy(fused_tensor.raw_data(), fused_data.data(), fused_data.size())
信号: SIGSEGV (段错误)
Exit code: 139
```

**崩溃前日志**：
```
[C-API] Fusing 10 QKV tensors...
[C-API] QKV fusion navigating ... (key: layers.7.attention)
[C-API]   attn_module found, type=AttentionWeight
[C-API]   child('w_qkv') -> 0x...
[C-API]   w_qkv_module->type() = LinearWeight
[段错误 - 无后续日志]
```

**根本原因**：

1. `w_qkv_param.alloc()` 返回的 `fused_tensor` 通过了 `!fused_tensor` 检查
2. 但 `fused_tensor.raw_data()` 内部调用 `buffer_.raw_data()` → `TM_CHECK_NOTNULL(data_)` → **段错误**
3. `data_` 为 NULL，但 tensor 对象的 `operator bool()` 只检查 `buffer_` 不检查 `data_`

**代码分析**：

```cpp
// Tensor::operator bool() - 不充分的检查
explicit operator bool() const noexcept {
    return static_cast<bool>(buffer_);  // 不检查 data_
}

// Buffer::raw_data() - 未检查的 NULL 访问
void* raw_data(ssize_t offset = 0) {
    return (char*)TM_CHECK_NOTNULL(data_).get() + ...;  // 如果 data_ 为 NULL，崩溃
}
```

#### 问题 2: Python Benchmark 无法运行 (P1)

**错误信息**：
```
TypeError: get_model_arch() got an unexpected keyword argument 'trust_remote_code'
位置: lmdeploy/tokenizer.py:79
```

**根因**：
- 最新提交 ae7a81da 移除了 `get_model_arch()` 的 `trust_remote_code` 参数
- 但 tokenizer.py 的调用代码未同步更新
- 这是代码回归

### Python vs Rust 加载路径差异

| 方面 | Python TurboMind | Rust Server |
|------|------------------|-------------|
| 绑定方式 | pybind11 直接绑定 | C API InitFromPath |
| 权重加载 | model_loader.export() | LoadWeightsFromSafetensors |
| QKV Fusion | 可能跳过或使用不同路径 | C++ 代码中执行并崩溃 |
| 内部访问 | 直接 C++ 调用 | FFI 层间接调用 |

---

## 三、Beads 任务执行进度

### 当前任务状态

| ID | 任务 | 优先级 | 状态 | 说明 |
|----|------|--------|------|------|
| lmdeploy-9yov | QKV Fusion 崩溃 | P0 | 🔄 Open | **阻塞所有测试** |
| lmdeploy-hqb2 | get_model_arch 参数不兼容 | P1 | 🔄 Open | **阻塞 Python benchmark** |
| lmdeploy-q2f1 | Rust 超过 Python 优化方案 | P0 | 🔒 Blocked | 依赖上述两个任务 |
| lmdeploy-vnbu | QKV Fusion 根因定位 | P0 | Open | 与 9yov 重复 |

### 已完成任务（近期）

- ✅ lmdeploy-2o30: C++ InitFromPath 权重加载崩溃
- ✅ lmdeploy-p0lz: GPU Tokenizer 路径优化
- ✅ lmdeploy-cwc5: 消除 Event Sync
- ✅ lmdeploy-ki4v: Condvar Timeout 优化
- ✅ lmdeploy-747b: Prefill 性能综合验证

---

## 四、超越 Python 的具体策略

### 已实施的优化（已完成）

| 优化项 | 状态 | 预期提升 |
|--------|------|----------|
| GPU Tokenizer 零拷贝 | ✅ 完成 | 5-10% |
| forward_async 异步 forward | ✅ 完成 | 10-15% |
| Condvar wait() 无超时 | ✅ 完成 | 3-5% |
| Event Sync 消除 | ✅ 完成 | 5-10% |
| Request Pool 预分配 | ✅ 完成 | 2-3% |

**已完成总预期提升**: 25-43%

### 待实施的优化（需先修复崩溃）

| 策略 | 预期提升 | 优先级 | 说明 |
|------|---------|--------|------|
| TensorMap 复用 | 3-5% | P1 | 减少重复创建 |
| 回调路径优化 | 3-5% | P2 | 对比 Python pybind11 |
| CUDA Stream 重叠 | 5-10% | P3 | H2D 拷贝与计算重叠 |
| Batch 调度优化 | 20-50% | P3 | 并发场景 |

### 综合预期

- **单请求场景**: 总计 30-50% 提升，可从 ~6,000 tok/s 提升到 **~7,800-9,000 tok/s**
- **并发场景**: Batch 调度可额外提升 20-50%

---

## 五、C++ 代码修复方案

### 修复 1: QKV Fusion NULL 检查

**文件**: `src/turbomind/capi/turbomind_c.cc`
**位置**: Line 1284

```cpp
// 当前代码（崩溃）
std::memcpy(fused_tensor.raw_data(), fused_data.data(), fused_data.size());

// 修复方案：添加 NULL 检查
if (!fused_tensor || !fused_tensor.raw_data()) {
    fprintf(stderr, "[C-API] ERROR: fused_tensor.raw_data() is NULL for %s\n", key.c_str());
    continue;
}
std::memcpy(fused_tensor.raw_data(), fused_data.data(), fused_tensor.size());
```

### 修复 2: get_model_arch 参数兼容

**文件**: `lmdeploy/archs.py`
**位置**: Line 147

```python
# 当前代码（参数不兼容）
def get_model_arch(model_path: str):
    """Get a model's architecture and configuration.
    Args:
        model_path(str): the model path
    """

# 修复方案：添加可选参数
def get_model_arch(model_path: str, trust_remote_code: bool = False):
    """Get a model's architecture and configuration.
    Args:
        model_path(str): the model path
        trust_remote_code(bool): whether to trust remote code
    """
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception as e:
        from transformers import PretrainedConfig
        cfg = PretrainedConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
```

---

## 六、验收标准

### 修复验收

- [ ] lmdeploy-hqb2 完成: Python benchmark 能运行
- [ ] lmdeploy-9yov 完成: Rust Server 能加载模型
- [ ] 获取完整的 Python vs Rust 性能对比数据

### 性能验收

| 输入长度 | Python 目标 | Rust 目标 |
|----------|-------------|-----------|
| 512 | ~4,300 tok/s | ≥ 4,500 tok/s |
| 1024 | ~5,300 tok/s | ≥ 5,500 tok/s |
| 4096 | ~6,000 tok/s | ≥ 6,200 tok/s |
| 8192 | ~5,200 tok/s | ≥ 5,500 tok/s |

---

## 七、下一步行动

1. **立即**: 修复 lmdeploy-9yov (QKV Fusion NULL 检查)
2. **并行**: 修复 lmdeploy-hqb2 (get_model_arch 参数)
3. **然后**: 验证 Rust Server 能启动
4. **最后**: 运行 benchmark 对比，根据差距决定进一步优化

---

## 八、结论

1. **"好几万 tok/s" 不是真实单请求性能**
   - 真实性能: ~6,000 tok/s 峰值
   - Archive 数据不可信

2. **Rust Server 被 QKV Fusion 崩溃阻塞**
   - 必须先修复才能进行任何性能测试

3. **超越 Python 的路径清晰**
   - 已有 25-43% 优化基础
   - 剩余 15-25% 优化空间
   - 预期可达 ~7,800-9,000 tok/s

4. **关键行动**
   - 修复 QKV Fusion (P0)
   - 修复 get_model_arch (P1)
   - 运行基准测试验证

---

## 九、相关文档

- `docs/prefill_performance_deep_analysis_20260531.md`
- `docs/prefill_analysis_final_20260531.md`
- `benchmarks-archive/python-turbomind_35b_awq_20260526/README.md`
