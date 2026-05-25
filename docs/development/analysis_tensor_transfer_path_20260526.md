# Tensor 传递路径完整分析

## 概览

本文档详细分析 `cpp_engine.rs` 中 input_ids 从 tokenizer 到 TurboMind C++ 引擎的完整内存流转路径。

---

## 路径 1: `set_input_ids` (CPU Path，最慢)

**调用链**: `Vec<u32>` → `INPUT_ID_BUFFER` (thread-local `Vec<i64>`) → `TM_TensorMap_SetInt64`

### 内存流转

```
┌────────────────────────────────────────────────────────────────────┐
│                        CPU PATH (Slowest)                          │
├────────────────────────────────────────────────────────────────────┤
│  tokenizer::encode_ids()                                           │
│       │                                                             │
│       ▼                                                             │
│  Vec<u32> (堆分配，来自 tokenizer)                                  │
│       │                                                             │
│       │  [COPY 1] + [TYPE CONVERSION]                              │
│       │  INPUT_ID_BUFFER.with() → extend(input_ids.map(|&id| id as i64)) │
│       │                                                             │
│       ▼                                                             │
│  INPUT_ID_BUFFER: Vec<i64> (thread-local，堆分配)                  │
│       │                                                             │
│       │  [COPY 2]                                                   │
│       │  TM_TensorMap_SetInt64(name, &buf, ndim, shape)            │
│       │  → C++ TensorMap::set() creates Tensor, copies data        │
│       │                                                             │
│       ▼                                                             │
│  C++ Tensor (CPU memory, i64)                                      │
│       │                                                             │
│       ▼                                                             │
│  TurboMind Forward (可能需要额外 H2D copy)                         │
└────────────────────────────────────────────────────────────────────┘
```

### 拷贝次数: **2次 CPU copy** + 潜在的 GPU 传输

| 拷贝点 | 类型 | 耗时因素 |
|--------|------|----------|
| u32→i64 转换 + 扩展 | CPU，内存带宽 | 1-5 μs (典型 <4K tokens) |
| TensorMap 内部拷贝 | CPU，内存带宽 | 1-5 μs |

---

## 路径 2: `set_input_ids_gpu` (GPU Path，同步传输)

**调用链**: `Vec<u32>` → `PINNED_HOST_BUFFER` (i64) → `GPU_INPUT_BUFFER` → `TM_TensorMap_SetInt64GPU`

### 内存流转

```
┌────────────────────────────────────────────────────────────────────┐
│                    GPU PATH - Synchronous                          │
├────────────────────────────────────────────────────────────────────┤
│  tokenizer::encode_ids()                                           │
│       │                                                             │
│       ▼                                                             │
│  Vec<u32> (堆分配)                                                  │
│       │                                                             │
│       │  [COPY 1] + [TYPE CONVERSION]                              │
│       │  pinned_slice.copy_from_slice(input_ids.iter().map(|&id| id as i64)) │
│       │                                                             │
│       ▼                                                             │
│  PINNED_HOST_BUFFER: *mut c_void (cudaMallocHost, i64)            │
│  │  Pinned memory enables faster DMA transfer                     │
│       │                                                             │
│       │  [COPY 2] - BLOCKING H2D Transfer                          │
│       │  cudaMemcpy(gpu_ptr, pinned_ptr, size, HostToDevice)       │
│       │  │  → CPU blocks until transfer complete                   │
│       │                                                             │
│       ▼                                                             │
│  GPU_INPUT_BUFFER: *mut c_void (cudaMalloc, i64)                   │
│       │                                                             │
│       │  [ZERO-COPY]                                                │
│       │  TM_TensorMap_SetInt64GPU(name, gpu_ptr, ndim, shape)      │
│       │  → C++ TensorMap::set() creates GPU Tensor, stores pointer  │
│       │                                                             │
│       ▼                                                             │
│  C++ Tensor (GPU memory, i64) - 直接使用 GPU 指针                  │
│       │                                                             │
│       ▼                                                             │
│  TurboMind Forward (GPU 计算直接使用此 Tensor)                      │
└────────────────────────────────────────────────────────────────────┘
```

### 拷贝次数: **2次 copy** (1 CPU + 1 H2D)

| 拷贝点 | 类型 | 耗时因素 | 带宽 |
|--------|------|----------|------|
| u32→i64 转换 + pinned 写入 | CPU | 循环转换 | ~10-50 GB/s |
| cudaMemcpy (Host→Device) | PCIe DMA | PCIe Gen4/5 带宽 | ~25-50 GB/s |

**总耗时估算** (对于 4096 tokens):
- u32→i64 转换: ~40 μs (假设 50 GB/s 有效带宽)
- H2D 传输: ~320 μs (假设 25 GB/s PCIe，4096 * 8 bytes = 32KB)

---

## 路径 3: `set_input_ids_gpu_async` (GPU Path，异步传输)

**调用链**: `Vec<u32>` → `PINNED_HOST_BUFFER` (i64) → `GPU_INPUT_BUFFER` → `TM_TensorMap_SetInt64GPU` (异步)

### 内存流转

```
┌────────────────────────────────────────────────────────────────────┐
│                    GPU PATH - Asynchronous                         │
├────────────────────────────────────────────────────────────────────┤
│  tokenizer::encode_ids()                                           │
│       │                                                             │
│       ▼                                                             │
│  Vec<u32> (堆分配)                                                  │
│       │                                                             │
│       │  [COPY 1] + [TYPE CONVERSION]                              │
│       │  pinned_slice.copy_from_slice(...) (同同步路径)            │
│       │                                                             │
│       ▼                                                             │
│  PINNED_HOST_BUFFER (pinned memory, i64)                           │
│       │                                                             │
│       │  [COPY 2] - NON-BLOCKING H2D Transfer                      │
│       │  cudaMemcpyAsync(gpu_ptr, pinned_ptr, size, HostToDevice, stream) │
│       │  │  → CPU returns immediately, transfer in background      │
│       │                                                             │
│       ▼                                                             │
│  GPU_INPUT_BUFFER (GPU memory, i64)                                │
│       │                                                             │
│       │  CudaEvent::record(stream) → 用于后续同步                  │
│       │                                                             │
│       │  [ZERO-COPY]                                                │
│       │  TM_TensorMap_SetInt64GPU(...) (同同步路径)                 │
│       │                                                             │
│       ▼                                                             │
│  C++ Tensor (GPU memory, i64)                                      │
│       │                                                             │
│       ▼                                                             │
│  TurboMind Forward (event 同步后使用)                              │
└────────────────────────────────────────────────────────────────────┘
```

### 优化点

**CPU 工作重叠**: 异步传输期间 CPU 可以继续处理其他任务（如其他请求的预处理）

**总耗时**: 隐藏 H2D 延迟，感知延迟降低 ~300 μs

---

## 路径 4: `set_input_ids_gpu_uint32_async` (优化路径，当前最优)

**调用链**: `Vec<u32>` → `PINNED_UINT32_BUFFER` (u32) → `GPU_UINT32_BUFFER` → `TM_TensorMap_SetDLPack` (uint32)

### 内存流转

```
┌────────────────────────────────────────────────────────────────────┐
│              OPTIMIZED UINT32 ASYNC PATH (Current Best)            │
├────────────────────────────────────────────────────────────────────┤
│  tokenizer::encode_ids()                                           │
│       │                                                             │
│       ▼                                                             │
│  Vec<u32> (堆分配)                                                  │
│       │                                                             │
│       │  [COPY 1] - NO TYPE CONVERSION ✨                          │
│       │  pinned_slice.copy_from_slice(input_ids)                   │
│       │  │  → u32 直接拷贝，无需转换                                │
│       │                                                             │
│       ▼                                                             │
│  PINNED_UINT32_BUFFER (pinned memory, u32)                         │
│  │  Size: N * 4 bytes (vs N * 8 bytes for i64)                    │
│       │                                                             │
│       │  [COPY 2] - NON-BLOCKING H2D Transfer                      │
│       │  cudaMemcpyAsync(gpu_ptr, pinned_ptr, size, ..., stream)   │
│       │  │  → 传输量减半 (4 bytes vs 8 bytes per token)            │
│       │                                                             │
│       ▼                                                             │
│  GPU_UINT32_BUFFER (GPU memory, u32)                               │
│       │                                                             │
│       │  CudaEvent::record(stream)                                 │
│       │                                                             │
│       │  [ZERO-COPY via DLPack] ✨                                 │
│       │  TM_TensorMap_SetDLPack(                                    │
│       │      "input_ids", gpu_ptr, shape,                          │
│       │      dl_type_code=4 (kDLUInt), dl_type_bits=32,            │
│       │      device_type=2 (kDLCUDA)                               │
│       │  )                                                          │
│       │  → C++ TensorMap::set_from_dlpack()                        │
│       │  → Creates Tensor with GPU pointer, NO internal copy       │
│       │                                                             │
│       ▼                                                             │
│  C++ Tensor (GPU memory, uint32)                                   │
│  │  TurboMind 内部处理 uint32 类型                                 │
│       │                                                             │
│       ▼                                                             │
│  TurboMind Forward (GPU 计算直接使用)                              │
└────────────────────────────────────────────────────────────────────┘
```

### 优化总结

| 优化项 | 路径 2/3 (i64) | 路径 4 (u32) | 收益 |
|--------|----------------|--------------|------|
| 类型转换 | 需要 (u32→i64) | 无 ✨ | CPU 循环节省 |
| Pinned buffer 大小 | N * 8 bytes | N * 4 bytes | 内存减半 |
| H2D 传输量 | N * 8 bytes | N * 4 bytes | 带宽减半 |
| C++ 拷贝 | SetInt64GPU (可能拷贝) | SetDLPack (zero-copy) | 零拷贝 ✨ |

**总耗时估算** (对于 4096 tokens):
- u32 拷贝到 pinned: ~16 μs (vs 40 μs for i64 conversion)
- H2D 传输: ~160 μs (vs 320 μs，数据量减半)

---

## Thread-Local Buffer 管理

### Buffer 初始化

```rust
thread_local! {
    static INPUT_ID_BUFFER: RefCell<Vec<i64>> = ...;      // CPU path
    static GPU_INPUT_BUFFER: RefCell<GpuInputIdsBuffer> = ...;   // GPU i64
    static PINNED_HOST_BUFFER: RefCell<PinnedHostBuffer> = ...; // Pinned i64
    static GPU_UINT32_BUFFER: RefCell<GpuUint32Buffer> = ...;   // GPU u32 ✨
    static PINNED_UINT32_BUFFER: RefCell<PinnedUint32Buffer> = ...; // Pinned u32 ✨
}
```

### Growth 策略

```rust
fn get_or_grow(&mut self, needed_elements: usize) -> *mut c_void {
    if needed_elements > self.capacity {
        let new_capacity = needed_elements.next_power_of_two()
            .max(self.capacity * 2);
        // Free old, allocate new
        cudaMalloc(...);
    }
    self.gpu_ptr
}
```

**特性**:
- 初始容量: 8192 元素
- 增长策略: `max(needed.next_power_of_two(), capacity * 2)`
- 生命周期: thread-local，随线程退出释放

---

## C++ 侧分析

### `SetTensorCommon<T>` (路径 2, 3)

```cpp
template<typename T>
void SetTensorCommon(TensorMap* map, const char* name,
                     const T* data, int ndim, const int64_t* shape,
                     turbomind::DeviceType device_type) {
    auto dtype = data_type_v<T>;  // 类型推导
    map->set(name, Tensor{data, shape, dtype, device_type});
}
```

**行为**: 创建 Tensor 对象，**可能**根据 `device_type` 进行内部拷贝

### `TM_TensorMap_SetDLPack` (路径 4) ✨

```cpp
void TM_TensorMap_SetDLPack(TM_TensorMap* map, const char* name,
                            const void* data, int ndim, const int64_t* shape,
                            int dl_type_code, int dl_type_bits, int device_type) {
    auto dtype = DLPackDtypeToTurbomind(dl_type_code, dl_type_bits);
    auto dl_device = static_cast<DLDeviceType>(device_type);
    map->set_from_dlpack(name, data, shape, dtype, dl_device);
}
```

**行为**: DLPack 协议设计为**零拷贝**，直接使用传入的 GPU 指针

---

## 拷贝点汇总表

| 路径 | 拷贝 1 | 拷贝 2 | C++ 侧 | 总拷贝 |
|------|--------|--------|--------|--------|
| 1. `set_input_ids` | u32→i64 (CPU) | Vec→Tensor (CPU) | 可能 H2D | 2-3 |
| 2. `set_input_ids_gpu` | u32→i64 (CPU→Pinned) | H2D (DMA) | 可能拷贝 | 2-3 |
| 3. `set_input_ids_gpu_async` | u32→i64 (CPU→Pinned) | H2D Async (DMA) | 可能拷贝 | 2-3 |
| 4. `set_input_ids_gpu_uint32_async` | u32 (CPU→Pinned) ✨ | H2D Async (DMA, 减半) | 零拷贝 ✨ | **2** |

---

## 性能估算

假设: 4096 tokens, PCIe Gen4 x16 (25 GB/s)

| 路径 | CPU 耗时 | H2D 耗时 | 总耗时 |
|------|----------|----------|--------|
| 1. CPU | ~40 μs | ~320 μs (隐式) | ~360 μs |
| 2. GPU 同步 | ~40 μs | ~320 μs (阻塞) | ~360 μs |
| 3. GPU 异步 | ~40 μs | ~320 μs (隐藏) | ~40 μs (感知) |
| 4. u32 异步 ✨ | ~16 μs | ~160 μs (隐藏) | **~16 μs (感知)** |

---

## 进一步优化方向

1. **Tokenizer 直接输出到 Pinned Memory**
   - 当前: Vec<u32> (堆) → Pinned Buffer
   - 优化: Tokenizer 直接写入 Pinned Buffer
   - 收益: 消除一次 CPU 拷贝

2. **CUDA Graph**
   - 对固定长度的请求捕获 kernel 序列
   - 收益: 消除 kernel launch 开销

3. **Pipeline 优化**
   - 在异步传输期间准备其他输入 (如 attention mask)
   - 收益: 更好的 CPU-GPU 重叠

4. **Direct Input from PyTorch**
   - 如果输入已在 GPU，使用 DLPack 直接传递
   - 收益: 完全零拷贝
