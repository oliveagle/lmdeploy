# 深度性能分析 - Rust Server Prefill 慢于 Python 的根因

**创建时间**: 2026-05-26
**Bead ID**: lmdeploy-gx7z
**优先级**: P0

---

## 问题描述

Rust Server 的 prefill 性能显著慢于 Python TurboMind：

| Context | Python TM | Rust Server | 差距 |
|---------|-----------|-------------|------|
| 1K      | 14,827 tok/s | 3,556 tok/s | **4.2x 慢** |
| 4K      | 33,727 tok/s | 3,409 tok/s | **9.9x 慢** |
| 8K      | 42,875 tok/s | 2,660 tok/s | **16.1x 慢** |

## 数据来源

- **Python 基线**: `BENCHMARK_PYTHON_TM_20260518.json` (2026-05-18 测试)
- **Rust 当前**: `lmdeploy-rust-server/tests/prefill_correct.json` (最新测试)

---

## 根因分析

### 1. Python 的 Tensor 传递路径

```python
# Python 路径 (lmdeploy/turbomind/turbomind.py)
input_ids = torch.IntTensor(input_ids)  # CPU tensor
inputs = dict(input_ids=input_ids)
inputs = _np_dict_to_tm_dict(inputs)    # TensorMap via DLPack

# _np_dict_to_tm_dict 实现
def _np_dict_to_tm_dict(np_dict: dict):
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)    # DLPack 转换
    return ret
```

**关键发现**: Python 传递的是 **CPU tensor**，C++ 内部会将其复制到 GPU。

### 2. Rust 的 Tensor 传递路径

```rust
// Rust 路径 (cpp_engine.rs)
let dlpack_input = DlpackInputTensor {
    name: "input_ids",
    data: gpu_tensor.gpu_ptr as *const c_void,  // GPU pointer!
    shape: vec![seq_len as i64],
    dtype: DlpackDtype::UInt(32),
    device: DlpackDevice::Cuda(0),              // CUDA device
};
set_tensor_from_dlpack(&mut input_tensors, &dlpack_input);
```

**关键发现**: Rust 直接传递 **GPU pointer** 给 C++，避免了 CPU→GPU 复制！

### 3. 矛盾点分析

理论上 Rust 应该更快（零拷贝 GPU 传递），但实际却慢很多。可能的原因：

#### a) 测量方法差异

- **Python**: TTFT 测量从 `stream_infer()` 调用开始到第一个 token
- **Rust**: 可能包含更多开销（gRPC、HTTP API 等）

#### b) C++ TensorMap 的内部处理

检查 `set_from_dlpack` 的 C++ 实现：

```cpp
// src/turbomind/python/bind.cpp
m.def("from_dlpack", [](py::object obj) {
    py::capsule cap = obj.attr("__dlpack__")();
    DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
    auto ret = DLManagedTensorToTritonTensor(dlmt);
    return ret;
});
```

DLPack tensor 被转换为内部 Tensor 对象。对于 GPU tensor，数据已经在 GPU 上。

#### c) Rust C API vs Python API

Rust 使用的是手动包装的 C API (`TM_TensorMap_SetFromDLPack`)，而 Python 使用 pybind11 绑定的对象方法。

可能存在差异：
1. **错误处理开销**: Rust FFI 调用可能更严格
2. **数据类型转换**: uint32_t vs int64_t 可能需要处理
3. **C++ 内部 GPU 拷贝**: 即使用 DLPack 传递 GPU pointer，C++ 内部可能仍会做某些处理

### 4. 性能瓶颈定位

#### a) Tokenization 开销

```rust
// tokenizer.rs - encode_to_gpu()
let raw_tokens = self.tokenizer.tokenizer.encode(text);  // 返回 Vec<u32>
```

splintr tokenizer 返回 `Vec<u32>`，这涉及：
- CPU 内存分配
- 数据复制
- 然后再复制到 pinned → GPU

#### b) DLPack 零拷贝的有效性

检查 C++ 是否真的使用 DLPack pointer 还是做了 copy：

```cpp
// 从 bind.cpp 看，from_dlpack 返回 Tensor
// Tensor 内部存储的是 DLPack tensor 的指针
// 如果 C++ 后续操作需要 CPU 访问，会触发隐式 copy
```

### 5. 验证建议

1. **直接比较 C++ API 调用**:
   - 使用 Python benchmark 脚本直接调用 C++ API
   - 绕过 Python pipeline 层，测量纯 C++ prefill 时间

2. **Rust C API 性能测试**:
   - 直接调用 `TM_TensorMap_SetFromDLPack`
   - 测量 tensor 传递开销

3. **DLPack 数据流验证**:
   - 在 C++ 层添加日志，确认数据是否真的在 GPU 上
   - 使用 `cudaMemcpyAsync` 的返回值验证

---

## 优化方向

### 短期 (已完成验证但需确认)

1. **DLPack 零拷贝** ✅ 已实现
   - GPU tensor 直接传递给 C++
   - 避免了 CPU→GPU 中间复制

2. **Thread-local buffer 复用** ✅ 已实现
   - 避免每次请求的内存分配

3. **Async H2D transfer** ✅ 已实现
   - 允许 CPU/GPU 并行工作

### 中期 (待验证)

1. **CUDA Graph**
   - 减少 kernel launch 开销
   - 对长序列更有效

2. **Prefix Caching**
   - 复用相同前缀的 KV cache
   - 对 chat 应用效果明显

3. **测量方法修正**
   - 确保 Rust 和 Python 使用相同的测量基准
   - 排除 gRPC/HTTP API 开销

---

## 结论

当前 Rust 性能大幅落后于 Python 的原因**不是 tensor 传递**（Rust 使用了零拷贝 GPU 传递）。

更可能的原因是：
1. **测量基准不一致**: Rust 包含 API 层开销
2. **C++ 引擎行为差异**: DLPack GPU pointer 在 C++ 内部可能有额外处理
3. **Tokenizer 效率**: splintr tokenizer vs HuggingFace tokenizer

建议下一步：
1. 使用纯 C++ benchmark（无 API 层）
2. 直接比较 Python 和 Rust 的 C++ API 调用时间
3. 验证 DLPack 数据流是否真正零拷贝

---

## 相关文件

- Python TurboMind: `lmdeploy/turbomind/turbomind.py`
- Rust C++ Engine: `lmdeploy-rust-server/src/model/cpp_engine.rs`
- Rust Tokenizer: `lmdeploy-rust-server/src/tokenizer.rs`
- C++ C API: `src/turbomind/capi/turbomind_c.cc`
- C++ Python Bind: `src/turbomind/python/bind.cpp`
- 分析文档: `docs/development/analysis_tensor_transfer_path_20260526.md`