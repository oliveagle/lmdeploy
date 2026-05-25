# Python vs Rust Tensor 传递机制对比

## 关键发现：Python 使用 DLPack 零拷贝

### Python TurboMind 调用链

```python
# turbomind.py:48-54
def _np_dict_to_tm_dict(np_dict: dict):
    """Map numpy.ndarray to turbomind's tensor."""
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)  # ← DLPack 零拷贝！
    return ret
```

**关键**：`_tm.from_dlpack(v)` 使用 **DLPack 协议**，这是**零拷贝**的。

### DLPack 零拷贝原理

```python
# Python 端
input_ids = torch.IntTensor(input_ids)  # CPU 上的 tensor
inputs = _np_dict_to_tm_dict(inputs)    # 通过 DLPack 传递给 C++

# C++ 端 (_turbomind.so)
// from_dlpack() 不复制数据，只是包装了指针
TensorMap inputs = ...;
// outputs = _tm_dict_to_torch_dict(outputs) 也是零拷贝
```

**DLPack**：
- 是一个跨框架的内存格式标准
- 允许不同框架（PyTorch、TensorFlow、JAX 等）共享**同一块内存**
- 只传递**指针 + 形状信息**，不复制数据
- Python `torch.Tensor` 和 C++ `_tm.Tensor` 共享底层内存

---

## Rust Server 调用链（有额外拷贝）

### Rust 端

```rust
// src/turbomind_engine.rs
fn create_tensor_map(input_ids: &[i32]) -> Result<TensorMap> {
    let mut map = TensorMap::new();
    let tensor = // 创建新的 Tensor
        Tensor::from_slice(input_ids)  // ← 拷贝数据！
        .with_dtype(DataType::Int32)?;
    map.insert("input_ids", tensor);
    Ok(map)
}
```

### 问题：Rust 到 C++ 的边界

```rust
// Rust Vec<i32>
let input_ids: Vec<i32> = ...;

// 传递给 C++ TensorMap
// 需要：
// 1. Rust Vec → C++ Tensor 包装器
// 2. 可能的内存布局转换
// 3. FFI 边界的数据对齐
```

**即使使用了 DLPack**：
- Rust 需要先创建 `torch.Tensor` 或 `ndarray`
- 然后通过 DLPack 传递给 C++
- 但 Rust → DLPack → C++ 的链路比 Python → DLPack → C++ 长

---

## 为什么 Python 不需要额外拷贝？

### 1. PyTorch 与 _turbomind.so 的天然集成

```python
# _turbomind.so 是用 pybind11 编译的
// pybind11 可以直接操作 Python 对象
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// C++ 代码可以直接访问 PyTorch tensor 的底层指针
Tensor from_dlpack(PyObject* obj) {
    // 获取 PyTorch tensor 的底层指针
    auto tensor = torch::utils::python_to_tensor(obj);
    // 包装成 TurboMind Tensor，零拷贝
    return Tensor::wrap(tensor.data_ptr(), ...);
}
```

### 2. 内存布局一致

- **PyTorch** 和 **TurboMind C++** 都使用相同的内存布局（行优先/列优先一致）
- 不需要转换内存格式
- 只需要包装指针

### 3. Python GIL 保护

```python
# Python 的 GIL 确保：
# 1. 在调用 C++ 期间，Python 不会修改 tensor
# 2. C++ 可以安全地访问 tensor 内存
inputs = _np_dict_to_tm_dict(inputs)
# 此时 GIL 释放，C++ 访问内存
```

---

## Rust Server 的问题

### 1. 多层 FFI 边界

```
Rust Vec<i32>
  ↓ (拷贝？)
Rust TensorWrapper
  ↓ (FFI 调用)
C++ TensorMap
  ↓ (DLPack？)
C++ TurboMind Tensor
```

### 2. 内存布局可能不同

- Rust 的 `Vec<i32>` 默认布局可能与 C++ 期望的不同
- 需要确保内存对齐和连续性

### 3. 生命周期管理

- Rust 的所有权系统 vs C++ 的原始指针
- 需要额外的安全检查和转换

---

## 解决方案

### 方案 1：Rust 直接使用 DLPack

```rust
// 使用 torch-sys 或 dlpack-rs
extern "C" {
    fn torch_from_dlpack(...);
}

// 在 Rust 中创建 PyTorch tensor，然后零拷贝传递
let tensor = torch::Tensor::of_blob(...);
let dlpack_tensor = tensor.to_dlpack();
// 传递给 C++
```

### 方案 2：预分配内存缓存

```rust
// 只在第一次分配，后续复用
struct TensorCache {
    input_buffer: Vec<i32>,
    output_buffer: Vec<i32>,
}

impl TensorCache {
    fn ensure_capacity(&mut self, size: usize) {
        if self.input_buffer.len() < size {
            self.input_buffer.resize(size, 0);
        }
    }
}
```

### 方案 3：Rust 直接调用 C++ TurboMind

绕过 `TurboMindCEngine`，直接调用 `model_inst.forward()`：
```rust
// 类似 Python 的方式
let outputs = model_inst.forward(
    &inputs,
    &session,
    &gen_cfg,
    stream_output,
);
```

---

## 总结

| 方面 | Python TurboMind | Rust Server |
|------|------------------|-------------|
| Tensor 传递 | DLPack 零拷贝 | 可能有多层拷贝 |
| 内存布局 | PyTorch ↔ C++ 一致 | Rust ↔ C++ 可能不一致 |
| FFI 层数 | Python → C++ (1 层) | Rust → C++ → C++ (2 层) |
| 生命周期 | GIL 保护 | Rust所有权 + C++ 原始指针 |

**Python 不需要拷贝的原因**：
1. PyTorch 和 TurboMind C++ 都使用相同的 DLPack 协议
2. `_turbomind.so` 用 pybind11 编译，可以直接访问 Python 对象
3. DLPack 只传递指针，不复制数据
4. 内存布局完全一致

**Rust 需要优化的方向**：
1. 使用 DLPack 或类似的零拷贝协议
2. 减少中间层，直接调用 C++ TurboMind
3. 预分配内存缓存，避免重复分配
