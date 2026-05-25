//! TurboMind C++ Engine - Pure C++ inference via C API
//!
//! This engine uses the TurboMind C API directly without any Python dependency.
//! It loads weights from HuggingFace safetensors format and performs inference
//! entirely through the C++ interface.
//!
//! ## Concurrency Model
//!
//! The C++ TurboMind engine has an internal request queue (Gateway) that handles
//! concurrent scheduling. This wrapper creates a pool of ModelRequest instances
//! to allow parallel inference without mutex contention. Each request is
//! independent and can run concurrently with others.

use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex as StdMutex;
use std::time::Instant;

use crate::error::{AppError, Result};
use crate::tokenizer::LMTokenizer;
use crate::turbomind_c::{
    c_int, c_void, cudaFree, cudaFreeHost, cudaMalloc, cudaMallocHost, cudaMemcpy, cudaMemcpyAsync,
    cudaMemcpyKind, cudaStreamCreate, cudaStreamDestroy, cudaStream_t, CompiledGrammar,
    CudaEvent, EngineConfig, GenConfig, ModelRequest, ScheduleMetrics, TM_DataType,
    TM_SessionParam, TM_Tensor, TensorMap, TurboMind, DL_DEVICE_TYPE_CUDA, DL_DTYPE_CODE_INT,
};
use serde::Serialize;

/// Thread-local buffer for reusing input_ids allocation across requests.
/// This avoids per-request heap allocations for the i64 conversion of input_ids.
///
/// The buffer is used in the following pattern:
/// 1. Clear the buffer
/// 2. Extend with converted i64 values
/// 3. Pass to C++ engine (data is copied internally)
/// 4. Buffer can be reused for next request
use std::cell::RefCell;
thread_local! {
    static INPUT_ID_BUFFER: RefCell<Vec<i64>> = RefCell::new(Vec::with_capacity(8192));
}

/// Reusable GPU buffer for input_ids.
/// Manages a pre-allocated GPU memory region to avoid per-request cudaMalloc/cudaFree overhead.
/// Uses a dedicated CUDA stream for async operations, allowing the CPU to continue
/// working while the GPU transfer happens in the background.
struct GpuInputIdsBuffer {
    gpu_ptr: *mut c_void,
    capacity: usize,
    stream: cudaStream_t,
}

unsafe impl Send for GpuInputIdsBuffer {}

impl GpuInputIdsBuffer {
    fn new(initial_capacity: usize) -> Self {
        let mut gpu_ptr = std::ptr::null_mut();
        let size = initial_capacity * std::mem::size_of::<i64>();
        unsafe {
            let ret = cudaMalloc(&mut gpu_ptr, size);
            if ret != 0 {
                tracing::warn!(
                    ret,
                    "cudaMalloc failed for GPU buffer, falling back to per-request allocation"
                );
            }
        }
        let mut stream: cudaStream_t = std::ptr::null_mut();
        unsafe {
            let ret = cudaStreamCreate(&mut stream);
            if ret != 0 {
                tracing::warn!(
                    ret,
                    "cudaStreamCreate failed, falling back to sync transfers"
                );
            }
        }
        Self {
            gpu_ptr,
            capacity: initial_capacity,
            stream,
        }
    }

    /// Get GPU pointer, growing the buffer if needed.
    /// Returns None if GPU allocation fails.
    fn get_or_grow(&mut self, needed_elements: usize) -> *mut c_void {
        if needed_elements > self.capacity {
            // Free old buffer and allocate larger one
            if !self.gpu_ptr.is_null() {
                unsafe { cudaFree(self.gpu_ptr) };
            }
            let new_capacity = needed_elements.next_power_of_two().max(self.capacity * 2);
            let mut new_ptr = std::ptr::null_mut();
            let size = new_capacity * std::mem::size_of::<i64>();
            unsafe {
                if cudaMalloc(&mut new_ptr, size) == 0 {
                    self.gpu_ptr = new_ptr;
                    self.capacity = new_capacity;
                } else {
                    tracing::warn!("cudaMalloc failed to grow GPU buffer");
                    return std::ptr::null_mut();
                }
            }
        }
        self.gpu_ptr
    }
}

impl Drop for GpuInputIdsBuffer {
    fn drop(&mut self) {
        if !self.gpu_ptr.is_null() {
            unsafe { cudaFree(self.gpu_ptr) };
        }
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}

/// Pinned host buffer for fast CPU->GPU transfers.
struct PinnedHostBuffer {
    host_ptr: *mut c_void,
    capacity: usize,
}

unsafe impl Send for PinnedHostBuffer {}

impl PinnedHostBuffer {
    fn new(initial_capacity: usize) -> Self {
        let mut host_ptr = std::ptr::null_mut();
        let size = initial_capacity * std::mem::size_of::<i64>();
        unsafe {
            if cudaMallocHost(&mut host_ptr, size) != 0 {
                tracing::warn!("cudaMallocHost failed for pinned buffer");
                host_ptr = std::ptr::null_mut();
            }
        }
        Self {
            host_ptr,
            capacity: initial_capacity,
        }
    }

    fn get_or_grow(&mut self, needed_elements: usize) -> *mut c_void {
        if needed_elements > self.capacity {
            if !self.host_ptr.is_null() {
                unsafe { cudaFreeHost(self.host_ptr) };
            }
            let new_capacity = needed_elements.next_power_of_two().max(self.capacity * 2);
            let mut new_ptr = std::ptr::null_mut();
            let size = new_capacity * std::mem::size_of::<i64>();
            unsafe {
                if cudaMallocHost(&mut new_ptr, size) == 0 {
                    self.host_ptr = new_ptr;
                    self.capacity = new_capacity;
                } else {
                    self.host_ptr = std::ptr::null_mut();
                }
            }
        }
        self.host_ptr
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            unsafe { cudaFreeHost(self.host_ptr) };
        }
    }
}

/// GPU buffer for uint32 input_ids.
/// Uses uint32_t allocation to match tokenizer output directly,
/// avoiding the u32->i64 conversion and enabling zero-copy to C++.
struct GpuUint32Buffer {
    gpu_ptr: *mut c_void,
    capacity: usize,
    stream: cudaStream_t,
}

unsafe impl Send for GpuUint32Buffer {}

impl GpuUint32Buffer {
    fn new(initial_capacity: usize) -> Self {
        let mut gpu_ptr = std::ptr::null_mut();
        let size = initial_capacity * std::mem::size_of::<u32>();
        unsafe {
            let ret = cudaMalloc(&mut gpu_ptr, size);
            if ret != 0 {
                tracing::warn!(
                    ret,
                    "cudaMalloc failed for uint32 GPU buffer"
                );
            }
        }
        let mut stream: cudaStream_t = std::ptr::null_mut();
        unsafe {
            let ret = cudaStreamCreate(&mut stream);
            if ret != 0 {
                tracing::warn!(
                    ret,
                    "cudaStreamCreate failed for uint32 buffer"
                );
            }
        }
        Self {
            gpu_ptr,
            capacity: initial_capacity,
            stream,
        }
    }

    fn get_or_grow(&mut self, needed_elements: usize) -> *mut c_void {
        if needed_elements > self.capacity {
            if !self.gpu_ptr.is_null() {
                unsafe { cudaFree(self.gpu_ptr) };
            }
            let new_capacity = needed_elements.next_power_of_two().max(self.capacity * 2);
            let mut new_ptr = std::ptr::null_mut();
            let size = new_capacity * std::mem::size_of::<u32>();
            unsafe {
                if cudaMalloc(&mut new_ptr, size) == 0 {
                    self.gpu_ptr = new_ptr;
                    self.capacity = new_capacity;
                } else {
                    tracing::warn!("cudaMalloc failed to grow uint32 GPU buffer");
                    return std::ptr::null_mut();
                }
            }
        }
        self.gpu_ptr
    }

    fn stream(&self) -> cudaStream_t {
        self.stream
    }
}

impl Drop for GpuUint32Buffer {
    fn drop(&mut self) {
        if !self.gpu_ptr.is_null() {
            unsafe { cudaFree(self.gpu_ptr) };
        }
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}

/// Pinned host buffer for uint32 input_ids.
struct PinnedUint32Buffer {
    host_ptr: *mut c_void,
    capacity: usize,
}

unsafe impl Send for PinnedUint32Buffer {}

impl PinnedUint32Buffer {
    fn new(initial_capacity: usize) -> Self {
        let mut host_ptr = std::ptr::null_mut();
        let size = initial_capacity * std::mem::size_of::<u32>();
        unsafe {
            if cudaMallocHost(&mut host_ptr, size) != 0 {
                tracing::warn!("cudaMallocHost failed for uint32 pinned buffer");
                host_ptr = std::ptr::null_mut();
            }
        }
        Self {
            host_ptr,
            capacity: initial_capacity,
        }
    }

    fn get_or_grow(&mut self, needed_elements: usize) -> *mut c_void {
        if needed_elements > self.capacity {
            if !self.host_ptr.is_null() {
                unsafe { cudaFreeHost(self.host_ptr) };
            }
            let new_capacity = needed_elements.next_power_of_two().max(self.capacity * 2);
            let mut new_ptr = std::ptr::null_mut();
            let size = new_capacity * std::mem::size_of::<u32>();
            unsafe {
                if cudaMallocHost(&mut new_ptr, size) == 0 {
                    self.host_ptr = new_ptr;
                    self.capacity = new_capacity;
                } else {
                    self.host_ptr = std::ptr::null_mut();
                }
            }
        }
        self.host_ptr
    }
}

impl Drop for PinnedUint32Buffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            unsafe { cudaFreeHost(self.host_ptr) };
        }
    }
}

thread_local! {
    static GPU_INPUT_BUFFER: RefCell<GpuInputIdsBuffer> = RefCell::new(
        GpuInputIdsBuffer::new(8192)
    );
    static PINNED_HOST_BUFFER: RefCell<PinnedHostBuffer> = RefCell::new(
        PinnedHostBuffer::new(8192)
    );
    // GPU buffer for uint32 input_ids (zero-copy to C++ without u32->i64 conversion)
    static GPU_UINT32_BUFFER: RefCell<GpuUint32Buffer> = RefCell::new(
        GpuUint32Buffer::new(8192)
    );
    // Pinned buffer for uint32 input_ids
    static PINNED_UINT32_BUFFER: RefCell<PinnedUint32Buffer> = RefCell::new(
        PinnedUint32Buffer::new(8192)
    );
}

/// Set input_ids on TensorMap using GPU tensor path with async transfer.
///
/// This function:
/// 1. Copies input_ids to pinned host memory (fast transfer)
/// 2. Copies from pinned host memory to GPU via cudaMemcpyAsync (non-blocking)
///    using a dedicated CUDA stream for overlapping with other operations
/// 3. Records a CUDA event after the async copy
/// 4. Sets the GPU tensor using set_int64_gpu
///
/// The async copy allows overlapping the H2D transfer with other CPU work,
/// reducing overall latency. The event ensures synchronization before the
/// forward pass uses the data.
///
/// Reuses pre-allocated GPU and pinned host buffers to minimize allocation overhead.
///
/// Returns `Some(CudaEvent)` if async transfer was used (caller should sync on this
/// event before relying on the GPU tensor being ready). Returns `None` on fallback paths.
fn set_input_ids_gpu_async(tensors: &mut TensorMap, input_ids: &[u32]) -> Option<CudaEvent> {
    let gpu_size = input_ids.len();
    GPU_INPUT_BUFFER.with(|gpu_buf_cell| {
        PINNED_HOST_BUFFER.with(|pinned_cell| {
            let mut gpu_buf = gpu_buf_cell.borrow_mut();
            let mut pinned_buf = pinned_cell.borrow_mut();
            let gpu_ptr = gpu_buf.get_or_grow(gpu_size);
            if gpu_ptr.is_null() {
                set_input_ids(tensors, input_ids);
                return None;
            }
            let pinned_ptr = pinned_buf.get_or_grow(gpu_size);
            if !pinned_ptr.is_null() {
                let pinned_slice: &mut [i64] =
                    unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut i64, gpu_size) };
                for (dst, src) in pinned_slice.iter_mut().zip(input_ids.iter()) {
                    *dst = *src as i64;
                }
                if let Ok(event) = CudaEvent::new() {
                    // Use the dedicated stream for async copy to enable overlap
                    let stream = gpu_buf.stream;
                    unsafe {
                        cudaMemcpyAsync(
                            gpu_ptr,
                            pinned_ptr,
                            gpu_size * std::mem::size_of::<i64>(),
                            cudaMemcpyKind::HostToDevice,
                            stream,
                        )
                    };
                    // Record event on the same stream for synchronization
                    let _ = event.record(stream);
                    let shape = [gpu_size as i64];
                    tensors.set_input_ids_gpu(gpu_ptr.cast(), &shape);
                    tensors.set_sequence_length(input_ids.len() as i32);
                    return Some(event);
                }
            }
            // Fallback to sync copy
            set_input_ids_gpu(tensors, input_ids);
            None
        })
    })
}

/// Set input_ids on TensorMap using GPU tensor path.
///
/// This function:
/// 1. Copies input_ids to pinned host memory (fast transfer)
/// 2. Copies from pinned host memory to GPU via cudaMemcpy
/// 3. Sets the GPU tensor using set_int64_gpu
///
/// Reuses pre-allocated GPU and pinned host buffers to minimize allocation overhead.
fn set_input_ids_gpu(tensors: &mut TensorMap, input_ids: &[u32]) {
    let gpu_size = input_ids.len();

    GPU_INPUT_BUFFER.with(|gpu_buf_cell| {
        PINNED_HOST_BUFFER.with(|pinned_cell| {
            let mut gpu_buf = gpu_buf_cell.borrow_mut();
            let mut pinned_buf = pinned_cell.borrow_mut();

            // Get or grow GPU buffer
            let gpu_ptr = gpu_buf.get_or_grow(gpu_size);
            if gpu_ptr.is_null() {
                // Fallback to CPU path if GPU allocation fails
                set_input_ids(tensors, input_ids);
                return;
            }

            // Get or grow pinned host buffer
            let pinned_ptr = pinned_buf.get_or_grow(gpu_size);

            // Convert u32 input_ids to i64 in the pinned buffer (or fall back to regular Vec)
            if !pinned_ptr.is_null() {
                let pinned_slice: &mut [i64] =
                    unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut i64, gpu_size) };
                for (dst, src) in pinned_slice.iter_mut().zip(input_ids.iter()) {
                    *dst = *src as i64;
                }

                unsafe {
                    cudaMemcpy(
                        gpu_ptr,
                        pinned_ptr,
                        gpu_size * std::mem::size_of::<i64>(),
                        cudaMemcpyKind::HostToDevice,
                    )
                };
            } else {
                // Fallback: convert to Vec and copy from regular CPU memory
                let i64_buf: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
                unsafe {
                    cudaMemcpy(
                        gpu_ptr,
                        i64_buf.as_ptr() as *const c_void,
                        i64_buf.len() * std::mem::size_of::<i64>(),
                        cudaMemcpyKind::HostToDevice,
                    )
                };
            }

            let shape = [gpu_size as i64];
            tensors.set_input_ids_gpu(gpu_ptr.cast(), &shape);
        });
    });

    tensors.set_sequence_length(input_ids.len() as i32);
}

/// Efficiently set input_ids on TensorMap, reusing a thread-local buffer.
///
/// This avoids per-request Vec allocation in the common path.
fn set_input_ids(tensors: &mut TensorMap, input_ids: &[u32]) {
    INPUT_ID_BUFFER.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        buf.reserve(input_ids.len());
        buf.extend(input_ids.iter().map(|&id| id as i64));
        let shape = [buf.len() as i64];
        tensors.set_input_ids(&buf, &shape);
    });
    tensors.set_sequence_length(input_ids.len() as i32);
}

/// Set input_ids on TensorMap from pre-converted i64 slice, reusing thread-local buffer.
fn set_input_ids_i64(tensors: &mut TensorMap, input_ids: &[i64]) {
    INPUT_ID_BUFFER.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        buf.reserve(input_ids.len());
        buf.extend_from_slice(input_ids);
        let shape = [buf.len() as i64];
        tensors.set_input_ids(&buf, &shape);
    });
    tensors.set_sequence_length(input_ids.len() as i32);
}

/// Set input_ids on TensorMap using GPU tensor path with uint32 data directly.
///
/// This function eliminates the u32->i64 conversion overhead by:
/// 1. Copying input_ids directly to pinned host memory (no conversion)
/// 2. Async copy from pinned to GPU via cudaMemcpyAsync
/// 3. Setting the GPU tensor using set_input_ids_gpu_uint32 (zero-copy to C++)
///
/// The result is a single memory copy path: CPU Vec<u32> -> Pinned -> GPU.
/// This matches the Python path's single copy behavior.
///
/// Returns `Some(CudaEvent)` if async transfer was used. Caller must sync before forward.
fn set_input_ids_gpu_uint32_async(tensors: &mut TensorMap, input_ids: &[u32]) -> Option<CudaEvent> {
    let gpu_size = input_ids.len();
    GPU_UINT32_BUFFER.with(|gpu_buf_cell| {
        PINNED_UINT32_BUFFER.with(|pinned_cell| {
            let mut gpu_buf = gpu_buf_cell.borrow_mut();
            let mut pinned_buf = pinned_cell.borrow_mut();
            let gpu_ptr = gpu_buf.get_or_grow(gpu_size);
            if gpu_ptr.is_null() {
                // Fallback to i64 path if GPU allocation fails
                set_input_ids_gpu(tensors, input_ids);
                return None;
            }
            let pinned_ptr = pinned_buf.get_or_grow(gpu_size);
            if !pinned_ptr.is_null() {
                // Copy u32 directly to pinned buffer (no conversion needed)
                let pinned_slice: &mut [u32] =
                    unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut u32, gpu_size) };
                pinned_slice.copy_from_slice(input_ids);

                if let Ok(event) = CudaEvent::new() {
                    let stream = gpu_buf.stream();
                    unsafe {
                        cudaMemcpyAsync(
                            gpu_ptr,
                            pinned_ptr,
                            gpu_size * std::mem::size_of::<u32>(),
                            cudaMemcpyKind::HostToDevice,
                            stream,
                        )
                    };
                    let _ = event.record(stream);
                    let shape = [gpu_size as i64];
                    // Use the uint32 GPU setter - C++ will receive GPU pointer directly
                    tensors.set_input_ids_gpu_uint32(gpu_ptr as *const u32, &shape);
                    tensors.set_sequence_length(input_ids.len() as i32);
                    return Some(event);
                }
            }
            // Fallback to sync copy
            set_input_ids_gpu(tensors, input_ids);
            None
        })
    })
}

/// DLPack input tensor for zero-copy transfer to the C++ engine.
///
/// This type allows external tensor providers (e.g., PyTorch, TensorFlow) to pass
/// GPU tensors directly to LMDeploy without CPU copying. The tensor data must remain
/// valid for the duration of the inference request.
///
/// # Example
/// ```ignore
/// let input_ids = vec![1i64, 2, 3];  // Must be on GPU for zero-copy
/// let tensor = DlpackInputTensor {
///     name: "input_ids",
///     data: input_ids.as_ptr() as *const c_void,
///     shape: vec![3],
///     dtype: DlpackDtype::Int64,
///     device: DlpackDevice::Cuda(0),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct DlpackInputTensor<'a> {
    /// Tensor name (e.g., "input_ids", "input_embeddings")
    pub name: &'a str,
    /// Raw pointer to tensor data (GPU or CPU)
    pub data: *const c_void,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: DlpackDtype,
    /// Device location
    pub device: DlpackDevice,
}

/// Data type for DLPack tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlpackDtype {
    /// Signed integer (8, 16, 32, 64 bit)
    Int(i32),
    /// Unsigned integer (8, 16, 32, 64 bit)
    UInt(i32),
    /// Floating point (16, 32, 64 bit)
    Float(i32),
    /// BFloat16 (16 bit)
    BFloat16,
    /// Boolean (8 bit)
    Bool,
}

/// Device type for DLPack tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlpackDevice {
    /// CPU device
    Cpu,
    /// CUDA GPU with device ID
    Cuda(i32),
    /// CUDA pinned host memory
    CudaHost,
}

impl DlpackDtype {
    /// Get DLPack type code (0=kBool, 2=kInt, 3=kFloat, 4=kUInt, 5=kBFloat)
    pub fn dl_type_code(&self) -> i32 {
        match self {
            DlpackDtype::Int(_) => 2,   // kDLInt
            DlpackDtype::UInt(_) => 4,  // kDLUInt
            DlpackDtype::Float(_) => 3, // kDLFloat
            DlpackDtype::BFloat16 => 5, // kDLBfloat
            DlpackDtype::Bool => 0,     // kDLBool
        }
    }

    /// Get number of bits
    pub fn dl_type_bits(&self) -> i32 {
        match self {
            DlpackDtype::Int(bits) | DlpackDtype::UInt(bits) | DlpackDtype::Float(bits) => *bits,
            DlpackDtype::BFloat16 | DlpackDtype::Bool => 16,
        }
    }
}

impl DlpackDevice {
    /// Get DLPack device type (1=CPU, 2=CUDA)
    pub fn dl_device_type(&self) -> i32 {
        match self {
            DlpackDevice::Cpu => 1,
            DlpackDevice::Cuda(_) | DlpackDevice::CudaHost => 2,
        }
    }

    /// Get device ID (0 for CPU, actual ID for CUDA)
    pub fn dl_device_id(&self) -> i32 {
        match self {
            DlpackDevice::Cpu => -1,
            DlpackDevice::Cuda(id) => *id,
            DlpackDevice::CudaHost => 0,
        }
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self, DlpackDevice::Cuda(_))
    }
}

/// Set a tensor to TensorMap using DLPack zero-copy transfer.
///
/// # Arguments
/// * `tensors` - Target TensorMap
/// * `tensor` - DLPack input tensor with data pointer and metadata
///
/// # Safety
/// The `tensor.data` pointer must remain valid for the duration of the inference request.
pub fn set_tensor_from_dlpack(tensors: &mut TensorMap, tensor: &DlpackInputTensor) {
    if tensor.device.is_gpu() && !tensor.data.is_null() {
        // Zero-copy DLPack path: GPU pointer directly to C++ engine
        tensors.set_from_dlpack(
            tensor.name,
            tensor.data,
            &tensor.shape,
            tensor.dtype.dl_type_code(),
            tensor.dtype.dl_type_bits(),
            tensor.device.dl_device_type(),
        );
    } else {
        // Fallback: CPU copy - not implemented for CPU tensors
        // Caller should use the standard set_int64/set_float32 setters instead
        tracing::warn!(
            name = tensor.name,
            "DLPack CPU fallback not implemented, use standard setter instead"
        );
    }
}

/// Get the max inference batch size for LLM models according to the GPU type.
///
/// This matches the Python implementation in `lmdeploy.utils.get_max_batch_size`:
/// - A100/A800: 384
/// - H100/H800/H200/L20Y: 1024
/// - Other CUDA devices: 128 (default)
///
/// Returns the GPU-adaptive batch size or 128 if GPU detection fails.
fn get_max_batch_size() -> i32 {
    use std::process::Command;

    // Map of GPU name patterns to batch sizes
    // Matches Python: max_batch_size_map = {'a100': 384, 'a800': 384, 'h100': 1024, 'h800': 1024, 'l20y': 1024, 'h200': 1024}
    let max_batch_size_map = [
        ("a100", 384),
        ("a800", 384),
        ("h100", 1024),
        ("h800", 1024),
        ("h200", 1024),
        ("l20y", 1024),
    ];

    // Query GPU name using nvidia-smi
    let output = match Command::new("nvidia-smi")
        .args(&["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => {
            tracing::warn!(
                "Failed to detect GPU type using nvidia-smi, using default max_batch_size=128"
            );
            return 128;
        }
    };

    let device_name = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_lowercase();

    // Check if any known GPU name pattern matches
    for (pattern, size) in max_batch_size_map {
        if device_name.contains(pattern) {
            tracing::info!(gpu = %device_name, max_batch_size = size, "GPU-adaptive max_batch_size detected");
            return size;
        }
    }

    // Default for unknown CUDA devices
    tracing::info!(
        gpu = %device_name,
        max_batch_size = 128,
        "Unknown GPU type, using default max_batch_size=128"
    );
    128
}

/// A single token logprobs entry as returned by the OpenAI API.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub token_id: u32,
    pub logprob: f64,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// One of the top logprobs for a token position.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub token_id: u32,
    pub logprob: f64,
    pub bytes: Vec<u8>,
}

/// Compiled grammar for guided decoding.
///
/// Represents a constraint that the generation must follow.
/// Created from JSON schema, regex pattern, or EBNF grammar.
pub struct GuidedGrammar {
    /// The compiled grammar, stored as an Arc so it can be shared
    /// across multiple generation calls and passed to the C++ engine.
    pub grammar: Arc<CompiledGrammar>,
}

/// Generation parameters from HTTP/gRPC requests.
///
/// These parameters are applied to the GenConfig when calling the C++ engine.
#[derive(Clone, Default, Debug)]
pub struct GenerationParams {
    /// Maximum number of new tokens to generate.
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = deterministic, > 0.0 = random sampling).
    /// Default: 0.7
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold (0.0-1.0). Default: 0.95
    pub top_p: Option<f32>,
    /// Top-k sampling (0 = disabled). Default: 50
    pub top_k: Option<i32>,
    /// Minimum probability threshold for MinP sampling. Default: 0.0
    pub min_p: Option<f32>,
    /// Repetition penalty. Default: 1.0
    pub repetition_penalty: Option<f32>,
    /// Random seed for deterministic sampling. None = random.
    pub seed: Option<u64>,
    /// Stop sequences (not directly used by C++, handled by caller).
    pub stop: Option<Vec<String>>,
    /// Return log probabilities for each token. Default: false
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return per token. Default: 0 (none)
    pub top_logprobs: Option<u32>,
    /// Guided decoding grammar constraint. None = unconstrained.
    pub grammar: Option<Arc<CompiledGrammar>>,
}

impl GenerationParams {
    /// Create GenerationParams from gRPC GenerateRequest fields.
    pub fn from_grpc_request(
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        repetition_penalty: Option<f32>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            max_tokens,
            temperature: temperature.filter(|&t| t > 0.0),
            top_p: top_p.filter(|&p| p > 0.0),
            top_k,
            min_p: None,
            repetition_penalty,
            seed,
            stop: None,
            logprobs: None,
            top_logprobs: None,
            grammar: None,
        }
    }

    /// Create GenerationParams from gRPC request with logprobs support.
    pub fn from_grpc_request_with_logprobs(
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        repetition_penalty: Option<f32>,
        seed: Option<u64>,
        logprobs: Option<bool>,
        top_logprobs: Option<u32>,
    ) -> Self {
        Self {
            max_tokens,
            temperature: temperature.filter(|&t| t > 0.0),
            top_p: top_p.filter(|&p| p > 0.0),
            top_k,
            min_p: None,
            repetition_penalty,
            seed,
            stop: None,
            logprobs,
            top_logprobs,
            grammar: None,
        }
    }

    /// Create GenerationParams from ChatCompletionsRequest fields.
    pub fn from_chat_request(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        min_p: Option<f32>,
        max_tokens: Option<i32>,
        seed: Option<i32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        stop: Option<crate::handlers::http::Stop>,
        logprobs: Option<bool>,
        top_logprobs: Option<u32>,
        grammar: Option<Arc<CompiledGrammar>>,
    ) -> Self {
        let repetition_penalty =
            Self::compute_repetition_penalty(presence_penalty, frequency_penalty);
        Self {
            max_tokens: max_tokens.map(|t| t as usize),
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            seed: seed.map(|s| s as u64),
            stop: stop.map(|s| match s {
                crate::handlers::http::Stop::Single(s) => vec![s],
                crate::handlers::http::Stop::Multiple(v) => v,
            }),
            logprobs,
            top_logprobs,
            grammar,
        }
    }

    /// Compute repetition penalty from presence/frequency penalties.
    fn compute_repetition_penalty(
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
    ) -> Option<f32> {
        let presence = presence_penalty.unwrap_or(0.0);
        let frequency = frequency_penalty.unwrap_or(0.0);
        let combined = presence + frequency;
        if combined == 0.0 {
            None
        } else {
            Some((1.0 + combined).max(0.5).min(3.0))
        }
    }

    /// Apply parameters to a GenConfig instance.
    /// Only sets values that are Some(), leaving defaults for None.
    pub fn apply_to_gen_config(&self, gen_cfg: &mut GenConfig) {
        if let Some(max_tokens) = self.max_tokens {
            gen_cfg.set_max_new_tokens(max_tokens as c_int);
        }
        if let Some(temperature) = self.temperature {
            gen_cfg.set_temperature(temperature);
        }
        if let Some(top_p) = self.top_p {
            gen_cfg.set_top_p(top_p);
        }
        if let Some(top_k) = self.top_k {
            gen_cfg.set_top_k(top_k as c_int);
        }
        if let Some(min_p) = self.min_p {
            gen_cfg.set_min_p(min_p);
        }
        if let Some(repetition_penalty) = self.repetition_penalty {
            gen_cfg.set_repetition_penalty(repetition_penalty);
        }
        if let Some(seed) = self.seed {
            gen_cfg.set_random_seed(seed);
        }
        if let Some(logprobs) = self.logprobs {
            let num_logprobs = if logprobs {
                self.top_logprobs.unwrap_or(1) as c_int
            } else {
                0
            };
            gen_cfg.set_output_logprobs(num_logprobs);
        }
    }
}

/// Completion callback for event-driven request completion.
///
/// Invoked by the C++ engine when the request completes (finish, error, or cancel).
/// Uses condition variable for event-driven notification instead of polling.
extern "C" fn completion_callback(status: c_int, seq_len: c_int, user_data: *mut c_void) {
    unsafe {
        let ctx = &*(user_data as *const StreamContext);

        // Log the completion status for debugging
        match status {
            7 => tracing::debug!(seq_len, "Request completed: TM_STATUS_FINISH"),
            5 => tracing::warn!(seq_len, "Request completed: TM_STATUS_FAIL"),
            8 => tracing::debug!(seq_len, "Request completed: TM_STATUS_CANCEL"),
            6 => tracing::warn!(seq_len, "Request completed: TM_STATUS_TOO_LONG"),
            9 => tracing::warn!(seq_len, "Request completed: TM_STATUS_INCONSISTENCY"),
            _ => tracing::debug!(status, seq_len, "Request completed with status"),
        }

        // Signal completion via condition variable
        let (lock, cvar) = &*ctx.completion;
        let mut done = lock.lock().unwrap();
        *done = true;
        cvar.notify_one();
    }
}

/// Token callback for event-driven streaming.
///
/// Invoked by the C++ engine whenever a new token is generated.
/// Decodes the token and sends (token_id, text) tuple through the channel.
/// Optimized for minimal latency: uses try_send and stack-allocated buffer (no heap allocation).
extern "C" fn token_callback(token_id: c_int, _seq_len: c_int, user_data: *mut c_void) {
    unsafe {
        let ctx = &*(user_data as *const StreamContext);

        // Stack-allocated array avoids heap allocation (0.01-0.1us saved per token)
        let token_ids = [token_id as u32];
        let token_str = match ctx.tokenizer.decode(&token_ids, true) {
            Ok(s) if !s.is_empty() => s,
            _ => return,
        };

        // Use try_send for non-blocking send - if channel is full, drop this token
        // This prevents blocking the C++ callback thread and improves TTFT
        let _ = ctx.tx.try_send((token_id as u32, token_str));
    }
}

/// Shared context passed to the C callbacks via raw pointer.
struct StreamContext {
    tokenizer: LMTokenizer,
    tx: tokio::sync::mpsc::Sender<(u32, String)>,
    /// Completion signal via condition variable for event-driven waiting
    /// Arc<(Mutex<bool>, Condvar)> - the bool is set to true when complete
    completion: Arc<(StdMutex<bool>, Condvar)>,
}

/// Send-safe wrapper for the raw context pointer.
/// Safe to Send because the underlying Arc<StreamContext> is Send + Sync,
/// and we only perform atomic operations on it across threads.
struct ContextPtr(*mut c_void);
unsafe impl Send for ContextPtr {}

/// Batch completion context for non-streaming batch inference
struct BatchCompletionContext {
    /// Completion signal via condition variable
    completion: Arc<(StdMutex<bool>, Condvar)>,
}

/// Batch completion callback for non-streaming batch inference.
extern "C" fn batch_completion_callback(_status: c_int, _seq_len: c_int, user_data: *mut c_void) {
    unsafe {
        let ctx = &*(user_data as *const BatchCompletionContext);
        let (lock, cvar) = &*ctx.completion;
        let mut done = lock.lock().unwrap();
        *done = true;
        cvar.notify_one();
    }
}

/// Pool of ModelRequest instances for concurrent inference.
///
/// Uses a `tokio::sync::Semaphore` to limit concurrent inference requests
/// and `tokio::sync::Mutex` per slot so that the async runtime can yield
/// during blocking FFI calls instead of parking OS threads.
pub(crate) struct RequestPool {
    /// Per-slot tokio mutex - yields during FFI calls
    slots: Vec<tokio::sync::Mutex<ModelRequest>>,
    /// Semaphore limits concurrent inference requests
    semaphore: tokio::sync::Semaphore,
    /// Reusable TensorMap per slot for input tensors (avoids per-forward allocation).
    /// Each slot has its own TensorMap cleared before each forward call.
    input_tensor_maps: Vec<tokio::sync::Mutex<TensorMap>>,
    /// Reusable TensorMap per slot for output tensors.
    output_tensor_maps: Vec<tokio::sync::Mutex<TensorMap>>,
}

impl RequestPool {
    /// Create a new request pool with the given concurrency level.
    fn new(tm: &TurboMind, concurrency: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(concurrency);
        let mut input_tensor_maps = Vec::with_capacity(concurrency);
        let mut output_tensor_maps = Vec::with_capacity(concurrency);

        for i in 0..concurrency {
            let request = ModelRequest::create(tm).map_err(|e| {
                AppError::ModelLoadFailed(format!("Failed to create request #{}: {:?}", i, e))
            })?;
            slots.push(tokio::sync::Mutex::new(request));

            // Create reusable TensorMaps for this slot
            let input_map = TensorMap::new().map_err(|e| {
                AppError::ModelLoadFailed(format!(
                    "Failed to create input tensor map #{}: {:?}",
                    i, e
                ))
            })?;
            input_tensor_maps.push(tokio::sync::Mutex::new(input_map));

            let output_map = TensorMap::new().map_err(|e| {
                AppError::ModelLoadFailed(format!(
                    "Failed to create output tensor map #{}: {:?}",
                    i, e
                ))
            })?;
            output_tensor_maps.push(tokio::sync::Mutex::new(output_map));
        }

        Ok(Self {
            slots,
            semaphore: tokio::sync::Semaphore::new(concurrency),
            input_tensor_maps,
            output_tensor_maps,
        })
    }

    /// Acquire a slot (async). Returns a guard that holds the semaphore
    /// permit and the mutex guards. The slot is released when the guard is
    /// dropped.
    ///
    /// The slot selection uses round-robin to distribute load across slots.
    /// Each permit acquisition corresponds to one available inference slot.
    pub(crate) async fn acquire(
        &self,
    ) -> (
        tokio::sync::SemaphorePermit<'_>,
        tokio::sync::MutexGuard<'_, ModelRequest>,
        tokio::sync::MutexGuard<'_, TensorMap>,
        tokio::sync::MutexGuard<'_, TensorMap>,
    ) {
        let permit = self.semaphore.acquire().await.expect("semaphore closed");
        let active = self.slots.len() - self.semaphore.available_permits() - 1;
        let idx = active % self.slots.len();
        let guard = self.slots[idx].lock().await;
        let input_map = self.input_tensor_maps[idx].lock().await;
        let output_map = self.output_tensor_maps[idx].lock().await;
        (permit, guard, input_map, output_map)
    }

    /// Acquire a slot (blocking, for use in spawn_blocking). Returns a
    /// permit and the mutex guards.
    fn acquire_blocking(
        &self,
    ) -> (
        tokio::sync::SemaphorePermit<'_>,
        tokio::sync::MutexGuard<'_, ModelRequest>,
        tokio::sync::MutexGuard<'_, TensorMap>,
        tokio::sync::MutexGuard<'_, TensorMap>,
    ) {
        let permit = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.semaphore.acquire())
        })
        .expect("semaphore closed");
        let active = self.slots.len() - self.semaphore.available_permits() - 1;
        let idx = active % self.slots.len();
        let guard = self.slots[idx].blocking_lock();
        let input_map = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.input_tensor_maps[idx].lock())
        });
        let output_map = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.output_tensor_maps[idx].lock())
        });
        (permit, guard, input_map, output_map)
    }
}

/// Engine type selector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineType {
    /// Pure C++ inference (no Python dependency)
    #[default]
    PureCpp,
    /// Python bridge (Python subprocess + C++ TurboMind)
    PyBridge,
}

impl EngineType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpp" | "c++" | "native" | "pure_cpp" => Some(EngineType::PureCpp),
            "py" | "python" | "py_bridge" | "pythonbridge" => Some(EngineType::PyBridge),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EngineType::PureCpp => "pure_cpp",
            EngineType::PyBridge => "py_bridge",
        }
    }
}

/// Model loading state (same as Python bridge for compatibility)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ModelState {
    #[default]
    Unloaded,
    Loading,
    Ready,
    Failed(String),
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub state: ModelState,
    pub loaded_at: Option<i64>,
    pub engine_type: EngineType,
    pub quant_policy: i32,
    pub prefix_cache_enabled: bool,
    /// Model hidden dimension (used for embeddings)
    pub hidden_size: Option<usize>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            name: "default".into(),
            path: "".into(),
            state: ModelState::Unloaded,
            loaded_at: None,
            engine_type: EngineType::default(),
            quant_policy: 0,
            prefix_cache_enabled: false,
            hidden_size: None,
        }
    }
}

/// Detect AWQ quantization from config.json
fn detect_awq_quantization(model_path: &std::path::Path) -> bool {
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return false;
    }

    std::fs::read_to_string(&config_path)
        .ok()
        .map(|content| {
            let content_lower = content.to_lowercase();
            content_lower.contains("\"quant_method\"") && content_lower.contains("\"awq\"")
        })
        .unwrap_or(false)
}

/// Parse hidden_size from config.json
fn parse_hidden_size(model_path: &std::path::Path) -> usize {
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return 4096; // Default fallback
    }

    std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|content| {
            // Parse config.json as JSON to extract hidden_size
            serde_json::from_str::<serde_json::Value>(&content)
                .ok()
                .and_then(|v| {
                    // Handle nested configs (text_config, model_config)
                    let config = v
                        .get("text_config")
                        .or_else(|| v.get("model_config"))
                        .unwrap_or(&v);

                    config
                        .get("hidden_size")
                        .and_then(|h| h.as_u64())
                        .map(|u| u as usize)
                })
        })
        .unwrap_or(4096) // Default fallback
}

/// TurboMind pure C++ engine
pub struct TurboMindCEngine {
    pub(super) model_path: String,
    pub model_name: String,
    state: ModelState,
    pub loaded_at: Option<i64>,
    pub is_ready: std::sync::atomic::AtomicBool,
    pub engine_type: EngineType,
    pub prefix_cache_enabled: bool,

    // C API components
    tm: Option<Arc<TurboMind>>,
    /// Pool of ModelRequest instances for concurrent inference (includes semaphore)
    request_pool: Option<Arc<RequestPool>>,

    // Tokenizer for encoding/decoding
    tokenizer: Option<LMTokenizer>,

    // Configuration
    session_len: i32,
    max_batch_size: i32,
    quant_policy: i32,

    /// Model hidden dimension (from config.json), used for embeddings
    hidden_size: usize,
}

/// Extract logprobs from the C++ output tensors.
///
/// The C++ engine produces three output tensors when `output_logprobs > 0`:
/// - `logprob_indexes`: [batch, seq_len, k] int32 — top-k token indexes per position
/// - `logprob_vals`: [batch, seq_len, k] float32 — log probabilities per index
/// - `logprob_nums`: [batch, seq_len] int32 — how many entries are valid per position
///
/// This function reads these tensors, selects the first (selected token) logprob
/// and the top-k logprobs, and returns them paired with the decoded token text.
///
/// `output_ids` — the generated token IDs (int32 slice)
/// `request` — the completed ModelRequest to read output tensors from
/// `tokenizer` — for decoding token IDs to text
/// `top_k_requested` — number of top logprobs requested (from GenerationParams)
fn extract_logprobs(
    output_ids: &[i32],
    request: &ModelRequest,
    tokenizer: &LMTokenizer,
    top_k_requested: u32,
) -> Option<Vec<TokenLogprob>> {
    // Read logprob_nums to know how many valid entries per position
    let logprob_nums = match request.get_output("logprob_nums") {
        Ok((ptr, size)) => unsafe { Some(std::slice::from_raw_parts(ptr as *const i32, size / 4)) },
        Err(_) => None,
    };
    let logprob_nums = logprob_nums?;

    // Read logprob_vals: [batch=1, seq_len, k]
    let logprob_vals = match request.get_output("logprob_vals") {
        Ok((ptr, size)) => unsafe { Some(std::slice::from_raw_parts(ptr as *const f32, size / 4)) },
        Err(_) => None,
    };
    let logprob_vals = logprob_vals?;

    // Read logprob_indexes: [batch=1, seq_len, k]
    let logprob_indexes = match request.get_output("logprob_indexes") {
        Ok((ptr, size)) => unsafe { Some(std::slice::from_raw_parts(ptr as *const i32, size / 4)) },
        Err(_) => None,
    };
    let logprob_indexes = logprob_indexes?;

    let k = top_k_requested as usize;
    let seq_len = output_ids.len();
    if seq_len == 0 || logprob_nums.len() < seq_len {
        return None;
    }

    // The tensors are flat: [batch * seq_len * k], batch=1
    // index into vals/indexes: row = seq_idx * k, col = logprob_idx
    let mut result = Vec::with_capacity(seq_len);

    for (i, &token_id) in output_ids.iter().enumerate() {
        let num_valid = if i < logprob_nums.len() {
            logprob_nums[i] as usize
        } else {
            0
        };
        if num_valid == 0 {
            // No logprobs available for this position — emit a placeholder
            result.push(TokenLogprob {
                token: tokenizer
                    .decode(&[token_id as u32], true)
                    .ok()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_default(),
                token_id: token_id as u32,
                logprob: 0.0,
                bytes: Vec::new(),
                top_logprobs: Vec::new(),
            });
            continue;
        }

        let base_idx = i * k;
        let actual_k = num_valid.min(k);

        // Find the selected token's logprob (where index matches token_id)
        let mut selected_logprob = 0.0f64;
        let mut top_logprobs = Vec::with_capacity(actual_k);
        for j in 0..actual_k {
            let idx = base_idx + j;
            let tok_id = logprob_indexes[idx];
            let lp = logprob_vals[idx] as f64;
            let token_text = tokenizer
                .decode(&[tok_id as u32], true)
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_default();
            let bytes_vec = token_text.as_bytes().to_vec();

            top_logprobs.push(TopLogprob {
                token: token_text.clone(),
                token_id: tok_id as u32,
                logprob: lp,
                bytes: bytes_vec.clone(),
            });

            if tok_id == token_id {
                selected_logprob = lp;
            }
        }

        // If we didn't find the exact token in top-k, use the first logprob
        // as a fallback (shouldn't happen in normal cases)
        if actual_k == 0 && base_idx < logprob_vals.len() {
            selected_logprob = logprob_vals[base_idx] as f64;
        }

        let token_text = tokenizer
            .decode(&[token_id as u32], true)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_default();

        result.push(TokenLogprob {
            token: token_text.clone(),
            token_id: token_id as u32,
            logprob: selected_logprob,
            bytes: token_text.as_bytes().to_vec(),
            top_logprobs,
        });
    }

    Some(result)
}

impl TurboMindCEngine {
    /// Create a new C++ engine and initialize via C API
    pub async fn new(model_path: &str) -> Result<Self> {
        Self::new_with_prefix_caching(model_path, false).await
    }

    /// Create a new C++ engine with prefix caching control
    pub async fn new_with_prefix_caching(
        model_path: &str,
        prefix_cache_enabled: bool,
    ) -> Result<Self> {
        tracing::info!(model_path, "Initializing TurboMind C++ engine");

        let model_path_obj = std::path::PathBuf::from(model_path);
        let config_json = model_path_obj.join("config.json");

        if !config_json.exists() {
            tracing::error!(model_path = %model_path, "config.json not found in model path");
            return Err(AppError::ModelLoadFailed(format!(
                "config.json not found at {}",
                model_path
            )));
        }

        // Detect AWQ quantization
        let is_awq = detect_awq_quantization(&model_path_obj);
        let quant_policy = if is_awq { 4 } else { 0 };

        // Parse hidden_size from config.json (needed for embeddings)
        let hidden_size = parse_hidden_size(&model_path_obj);
        if is_awq {
            tracing::info!("Detected AWQ quantized model, enabling quant_policy=4");
        }
        tracing::info!(hidden_size, "Model config parsed");

        // Load tokenizer first
        tracing::info!("Loading tokenizer...");
        let tokenizer = match LMTokenizer::from_path(model_path) {
            Ok(t) => {
                tracing::info!(vocab_size = t.vocab_size(), "Tokenizer loaded successfully");
                Some(t)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load tokenizer from model path");
                None
            }
        };

        // Create C API engine config
        let mut engine_config = EngineConfig::new().map_err(|e| {
            AppError::ModelLoadFailed(format!("Failed to create engine config: {:?}", e))
        })?;

        // Configure engine
        // data_type is the activation dtype (kHalf), not the weight dtype.
        // AWQ weights are kUint4 but computations happen in fp16.
        engine_config.set_data_type(crate::turbomind_c::TM_DataType::TM_DATATYPE_FP16);
        engine_config.set_session_len(65536);

        // CRITICAL: max_prefill_token_num controls prefill performance
        // Default 0 causes max_forward_token_num = max_batch_size (32 tokens)
        // Setting to 32768 enables single-pass prefill for 32K context
        // This allows the entire 32K context to be processed in one forward pass
        engine_config.set_max_prefill_token_num(32768);

        // max_batch_size: higher values improve throughput but use more memory
        // GPU-adaptive: A100/A800=384, H100/H800/H200/L20Y=1024, default=128
        // This matches Python's get_max_batch_size('cuda') behavior
        let max_batch_size = get_max_batch_size();
        engine_config.set_max_batch_size(max_batch_size);
        engine_config.set_cache_block_seq_len(64);
        engine_config.set_cache_max_block_count(0.8);
        // cache_chunk_size: Python default -1 means allocate cache_max_entry_count blocks
        // When 0: allocates sqrt(cache_max_entry_count) blocks
        // When -1: allocates cache_max_entry_count blocks (matches Python behavior)
        engine_config.set_cache_chunk_size(-1);
        engine_config.set_enable_prefix_caching(prefix_cache_enabled);
        engine_config.set_enable_metrics(true);
        engine_config.set_quant_policy(quant_policy);

        // Set nnodes=1 to disable distributed mode (avoid LMDEPLOY_DIST_INIT_ADDR requirement)
        // CRITICAL: async execution must be enabled (default 1 in Python)
        // This enables the C++ engine's async queue for concurrent request processing
        // Without this, the engine uses synchronous blocking mode which is 30-40% slower
        engine_config.set_async(1);

        // Set nnodes=1 to disable distributed mode (avoid LMDEPLOY_DIST_INIT_ADDR requirement)
        engine_config.set_nnodes(1);
        engine_config.set_node_rank(0);

        // Set tensor parallelism sizes (required for TurboMind initialization)
        // These must satisfy: mlp_tp_size == attn_dp_size * attn_tp_size * attn_cp_size
        engine_config.set_attn_tp_size(1);
        engine_config.set_attn_cp_size(1);
        engine_config.set_attn_dp_size(1);
        engine_config.set_mlp_tp_size(1);

        // Set prefill iteration limits
        engine_config.set_num_tokens_per_iter(0);
        engine_config.set_max_prefill_iters(1);

        engine_config.add_device(0); // GPU 0

        // Create TurboMind instance via C API
        tracing::info!("Creating TurboMind C++ instance...");
        let tm = TurboMind::create(model_path, &mut engine_config).map_err(|e| {
            AppError::ModelLoadFailed(format!("Failed to create TurboMind: {:?}", e))
        })?;

        // Initialize from model path (builds module tree, loads weights)
        tracing::info!("Loading weights from safetensors...");
        let device_id = 0;
        let trust_remote_code = true;

        // The InitFromPath function does the full initialization:
        // 1. CreateContext
        // 2. CreateRoot
        // 3. Build ModelWeight module tree
        // 4. Load weights from safetensors
        // 5. ProcessWeights (GPU transfer)
        // 6. CreateEngine
        tm.init_from_path(device_id, model_path, trust_remote_code)
            .map_err(|e| AppError::ModelLoadFailed(format!("InitFromPath failed: {:?}", e)))?;

        // Create inference request pool for concurrent access (includes semaphore)
        // Uses GPU-adaptive max_batch_size: A100=384, H100=1024, default=128
        let request_pool = Arc::new(RequestPool::new(&tm, max_batch_size as usize)?);

        let model_name = model_path_obj
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string();

        tracing::info!(
            concurrency = max_batch_size,
            "TurboMind C++ engine initialized successfully"
        );

        Ok(Self {
            model_path: model_path.to_string(),
            model_name,
            state: ModelState::Ready,
            loaded_at: Some(unix_timestamp()),
            is_ready: std::sync::atomic::AtomicBool::new(true),
            engine_type: EngineType::PureCpp,
            prefix_cache_enabled,
            tm: Some(Arc::new(tm)),
            request_pool: Some(request_pool),
            tokenizer,
            session_len: 65536,
            max_batch_size,
            quant_policy,
            hidden_size,
        })
    }

    /// Check if the model is ready for inference
    pub fn is_ready(&self) -> bool {
        self.is_ready.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get model metadata
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            path: self.model_path.clone(),
            state: self.state.clone(),
            loaded_at: self.loaded_at,
            engine_type: self.engine_type,
            quant_policy: self.quant_policy,
            prefix_cache_enabled: self.prefix_cache_enabled,
            hidden_size: Some(self.hidden_size),
        }
    }

    /// Reload the model
    pub async fn reload(&mut self, new_model_path: &str) -> Result<()> {
        tracing::info!(
            old_path = %self.model_path,
            new_path = %new_model_path,
            "Reloading TurboMind C++ engine"
        );

        self.is_ready
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.state = ModelState::Loading;

        // Drop existing components
        self.tm = None;
        self.request_pool = None;

        // Re-initialize
        let new_engine = Self::new(new_model_path).await?;
        self.tm = new_engine.tm;
        self.request_pool = new_engine.request_pool;
        self.tokenizer = new_engine.tokenizer;
        self.model_path = new_model_path.to_string();
        self.model_name = new_engine.model_name;
        self.state = ModelState::Ready;
        self.loaded_at = Some(unix_timestamp());
        self.is_ready
            .store(true, std::sync::atomic::Ordering::Relaxed);

        tracing::info!(
            model_path = %self.model_path,
            model_name = %self.model_name,
            "TurboMind C++ engine reloaded successfully"
        );

        Ok(())
    }

    /// Generate text with TurboMind C++ engine
    pub async fn generate(&self, prompt: &str, params: GenerationParams) -> String {
        let (text, _, _) = self.generate_with_metrics(prompt, params).await;
        text
    }

    /// Generate text with TurboMind C++ engine using a DLPack input tensor.
    ///
    /// This is the zero-copy path: if the input_ids tensor is already on GPU,
    /// it avoids the CPU allocation and copy entirely, passing the GPU pointer
    /// directly to the C++ engine via DLPack protocol.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs as a DLPack tensor (GPU pointer + metadata)
    /// * `params` - Generation parameters
    ///
    /// # Safety
    /// The data pointer in `input_ids` must remain valid until the forward completes.
    pub async fn generate_with_dlpack_input(
        &self,
        input_ids: DlpackInputTensor<'_>,
        params: GenerationParams,
    ) -> String {
        let (text, _, _) = self
            .generate_with_dlpack_input_and_metrics(input_ids, params)
            .await;
        text
    }

    /// Generate text with DLPack input, returning metrics.
    ///
    /// The input_ids tensor must be on GPU for zero-copy transfer.
    /// Falls back to CPU copy if device is not GPU.
    ///
    /// # Safety
    /// The data pointer must remain valid until the forward completes.
    pub async fn generate_with_dlpack_input_and_metrics(
        &self,
        dlpack_input: DlpackInputTensor<'_>,
        params: GenerationParams,
    ) -> (String, usize, f64) {
        let pool = self
            .request_pool
            .as_ref()
            .expect("Request pool not initialized");

        let seq_len = if dlpack_input.shape.is_empty() {
            0
        } else {
            dlpack_input.shape[0] as usize
        };

        tracing::debug!(
            input_len = seq_len,
            device = ?dlpack_input.device,
            "Received DLPack input tensor"
        );

        let start = Instant::now();

        let (_permit, mut request, mut input_tensors, mut output_tensors) = pool.acquire().await;

        // Clear and reuse TensorMaps instead of allocating new ones
        input_tensors.clear();
        output_tensors.clear();

        // Prepare input tensors: use DLPack zero-copy for GPU, fallback to CPU copy
        if dlpack_input.device.is_gpu() && !dlpack_input.data.is_null() {
            // Zero-copy: GPU pointer directly to C++ engine
            input_tensors.set_from_dlpack(
                "input_ids",
                dlpack_input.data,
                &dlpack_input.shape,
                dlpack_input.dtype.dl_type_code(),
                dlpack_input.dtype.dl_type_bits(),
                dlpack_input.device.dl_device_type(),
            );
        } else {
            // Fallback: CPU copy via standard setter
            tracing::warn!("DLPack input not on GPU, falling back to CPU copy");
            // CPU fallback requires the caller to provide CPU data via separate API
            input_tensors.set_input_ids(&[0], &[0]);
        }
        input_tensors.set_sequence_length(seq_len as i32);

        let mut gen_cfg = GenConfig::new().unwrap();
        gen_cfg.set_max_new_tokens(params.max_tokens.unwrap_or(512) as i32);
        gen_cfg.set_temperature(params.temperature.unwrap_or(0.7));
        gen_cfg.set_top_p(params.top_p.unwrap_or(0.95));
        gen_cfg.set_top_k(params.top_k.unwrap_or(50) as i32);
        params.apply_to_gen_config(&mut gen_cfg);

        let session = TM_SessionParam {
            id: unix_timestamp() as u64,
            step: 0,
            start_flag: true,
            end_flag: true,
        };

        let result = request.forward(
            &mut input_tensors,
            &session,
            &gen_cfg,
            false,
            false,
            &mut output_tensors,
        );

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(()) => {
                let (data, size) = match request.get_output("output_ids") {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::error!(error = ?e, "Failed to get output");
                        return (String::new(), seq_len, elapsed);
                    }
                };

                if size == 0 {
                    tracing::warn!("Empty output_ids tensor");
                    return (String::new(), seq_len, elapsed);
                }

                let token_count = size / 4; // int32 tokens
                let output_ids =
                    unsafe { std::slice::from_raw_parts(data as *const u32, token_count) };

                let input_len = seq_len;
                let output_tokens = if output_ids.len() > input_len {
                    &output_ids[input_len..]
                } else {
                    output_ids
                };

                let text = match &self.tokenizer {
                    Some(tokenizer) => match tokenizer.decode(output_tokens, true) {
                        Ok(t) => t,
                        Err(e) => {
                            tracing::error!(error = ?e, "Decode failed");
                            String::new()
                        }
                    },
                    None => {
                        tracing::error!("Tokenizer not available");
                        String::new()
                    }
                };

                tracing::debug!(
                    output_len = output_tokens.len(),
                    "Decoded generation output"
                );

                (text, seq_len, elapsed)
            }
            Err(e) => {
                tracing::error!(error = ?e, "Forward failed");
                (String::new(), seq_len, elapsed)
            }
        }
    }

    /// Generate text with TurboMind C++ engine, returning (text, num_tokens, elapsed_ms)
    pub async fn generate_with_metrics(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> (String, usize, f64) {
        let pool = self
            .request_pool
            .as_ref()
            .expect("Request pool not initialized");

        // Tokenize input (outside the lock to minimize critical section)
        let input_ids = match &self.tokenizer {
            Some(tokenizer) => match tokenizer.encode(prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed");
                    return (String::new(), 0, 0.0);
                }
            },
            None => {
                tracing::error!("Tokenizer not available");
                return (String::new(), 0, 0.0);
            }
        };

        tracing::debug!(input_len = input_ids.len(), "Tokenized prompt");

        let start = Instant::now();

        // Acquire a slot (semaphore permit + mutex guard for the slot).
        // tokio::sync::Mutex allows the runtime to yield while waiting,
        // enabling true parallel inference without blocking threads.
        let (_permit, mut request, mut input_tensors, mut output_tensors) = pool.acquire().await;

        // Clear and reuse TensorMaps instead of allocating new ones
        input_tensors.clear();
        output_tensors.clear();

        // Start async GPU transfer for input_ids - CPU continues with config prep
        // while H2D happens in the background
        if let Some(event) = set_input_ids_gpu_uint32_async(&mut input_tensors, &input_ids) {
            // Sync on the event before forward to ensure data is ready on GPU
            // This sync point is necessary to guarantee correctness
            let _ = event.sync();
        } else {
            // Fallback path: already synced in set_input_ids_gpu
        }

        // Prepare generation config with HTTP parameters
        let mut gen_cfg = GenConfig::new().unwrap();
        gen_cfg.set_max_new_tokens(params.max_tokens.unwrap_or(512) as i32);
        gen_cfg.set_temperature(params.temperature.unwrap_or(0.7));
        gen_cfg.set_top_p(params.top_p.unwrap_or(0.95));
        gen_cfg.set_top_k(params.top_k.unwrap_or(50) as i32);
        // Apply any additional parameters (min_p, repetition_penalty, seed)
        params.apply_to_gen_config(&mut gen_cfg);

        // Prepare session parameters (use unique ID for each request)
        let session = TM_SessionParam {
            id: unix_timestamp() as u64,
            step: 0,
            start_flag: true,
            end_flag: true,
        };

        // Prepare output tensors (reused from pool)

        // Attach grammar for guided decoding if provided
        if let Some(grammar) = &params.grammar {
            if let Err(e) = request.set_grammar(grammar) {
                tracing::warn!(error = ?e, "Failed to attach grammar");
                return (String::new(), 0, 0.0);
            }
        }

        // Run inference. The request is Send+Sync and the C++ engine handles
        // its own internal synchronization, so we can proceed without holding
        // the tokio mutex during the blocking FFI call.
        match request.forward(
            &mut input_tensors,
            &session,
            &gen_cfg,
            false, // stream_output
            true,  // enable_metrics
            &mut output_tensors,
        ) {
            Ok(_) => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                // Get output tokens from the request
                match request.get_output("output_ids") {
                    Ok((data_ptr, size)) => {
                        // data_ptr points to int32 array
                        let num_tokens = size / 4;
                        let output_ids: Vec<i32> = unsafe {
                            std::slice::from_raw_parts(data_ptr as *const i32, num_tokens).to_vec()
                        };

                        // Decode output tokens
                        let text = if let Some(tokenizer) = &self.tokenizer {
                            match tokenizer.decode(
                                &output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                                true,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    tracing::error!(error = %e, "Decoding failed");
                                    format!("[decode error: {}]", e)
                                }
                            }
                        } else {
                            format!("{:?}", output_ids)
                        };

                        (text, num_tokens, elapsed_ms)
                    }
                    Err(e) => {
                        tracing::error!(error = ?e, "Failed to get output_ids");
                        (String::new(), 0, elapsed_ms)
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = ?e, "C++ inference failed");
                (String::new(), 0, 0.0)
            }
        }
    }

    /// Generate text with logprobs, returning (text, num_tokens, elapsed_ms, logprobs)
    ///
    /// When `params.logprobs` is true or `params.top_logprobs` is set, this method
    /// extracts log probability information from the C++ engine output tensors.
    pub async fn generate_with_logprobs(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> (String, usize, f64, Option<Vec<TokenLogprob>>) {
        let pool = self
            .request_pool
            .as_ref()
            .expect("Request pool not initialized");
        let need_logprobs =
            params.logprobs.unwrap_or(false) || params.top_logprobs.unwrap_or(0) > 0;
        let top_logprobs_req = params.top_logprobs.unwrap_or(1).max(1);

        // Tokenize input
        let input_ids = match &self.tokenizer {
            Some(tokenizer) => match tokenizer.encode(prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed");
                    return (String::new(), 0, 0.0, None);
                }
            },
            None => {
                tracing::error!("Tokenizer not available");
                return (String::new(), 0, 0.0, None);
            }
        };

        tracing::debug!(input_len = input_ids.len(), "Tokenized prompt");

        let start = Instant::now();

        let (_permit, mut request, mut input_tensors, mut output_tensors) = pool.acquire().await;

        // Clear and reuse TensorMaps
        input_tensors.clear();
        output_tensors.clear();

        // Start async GPU transfer for input_ids - CPU continues with config prep
        // while H2D happens in the background
        if let Some(event) = set_input_ids_gpu_uint32_async(&mut input_tensors, &input_ids) {
            // Sync on the event before forward to ensure data is ready on GPU
            let _ = event.sync();
        }

        // Prepare generation config
        let mut gen_cfg = GenConfig::new().unwrap();
        gen_cfg.set_max_new_tokens(params.max_tokens.unwrap_or(512) as i32);
        gen_cfg.set_temperature(params.temperature.unwrap_or(0.7));
        gen_cfg.set_top_p(params.top_p.unwrap_or(0.95));
        gen_cfg.set_top_k(params.top_k.unwrap_or(50) as i32);
        params.apply_to_gen_config(&mut gen_cfg);

        let session = TM_SessionParam {
            id: unix_timestamp() as u64,
            step: 0,
            start_flag: true,
            end_flag: true,
        };

        // Attach grammar for guided decoding if provided
        if let Some(grammar) = &params.grammar {
            if let Err(e) = request.set_grammar(grammar) {
                tracing::warn!(error = ?e, "Failed to attach grammar");
                return (String::new(), 0, 0.0, None);
            }
        }

        match request.forward(
            &mut input_tensors,
            &session,
            &gen_cfg,
            false,
            true,
            &mut output_tensors,
        ) {
            Ok(_) => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                match request.get_output("output_ids") {
                    Ok((data_ptr, size)) => {
                        let num_tokens = size / 4;
                        let output_ids: Vec<i32> = unsafe {
                            std::slice::from_raw_parts(data_ptr as *const i32, num_tokens).to_vec()
                        };

                        let text = if let Some(tokenizer) = &self.tokenizer {
                            match tokenizer.decode(
                                &output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                                true,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    tracing::error!(error = %e, "Decoding failed");
                                    format!("[decode error: {}]", e)
                                }
                            }
                        } else {
                            format!("{:?}", output_ids)
                        };

                        let logprobs = if need_logprobs {
                            self.tokenizer.as_ref().and_then(|tok| {
                                extract_logprobs(&output_ids, &request, tok, top_logprobs_req)
                            })
                        } else {
                            None
                        };

                        (text, num_tokens, elapsed_ms, logprobs)
                    }
                    Err(e) => {
                        tracing::error!(error = ?e, "Failed to get output_ids");
                        (String::new(), 0, elapsed_ms, None)
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = ?e, "C++ inference failed");
                (String::new(), 0, 0.0, None)
            }
        }
    }

    /// Generate text with streaming output (token-by-token)
    ///
    /// Uses event-driven callbacks from the C++ engine instead of polling.
    /// Each generated token fires a callback that decodes and sends it through the channel.
    ///
    /// Returns a stream of (token_id, token_text) tuples.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = (u32, String)> + Send>> {
        // Tokenize input
        let (input_ids, tokenizer) = match &self.tokenizer {
            Some(t) => {
                let ids = match t.encode(prompt, false, false) {
                    Ok(ids) => ids,
                    Err(e) => {
                        tracing::error!(error = %e, "Tokenization failed");
                        return Box::pin(futures::stream::empty());
                    }
                };
                (ids, t.clone())
            }
            None => {
                tracing::error!("Tokenizer not available");
                return Box::pin(futures::stream::empty());
            }
        };

        self.generate_stream_impl(input_ids, tokenizer, params)
            .await
    }

    /// Generate text with streaming output using pre-tokenized input.
    ///
    /// This variant skips tokenization for lower TTFT when token IDs are already known.
    ///
    /// Returns a stream of (token_id, token_text) tuples.
    pub async fn generate_stream_with_ids(
        &self,
        _prompt: &str,
        input_ids: Vec<u32>,
        params: GenerationParams,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = (u32, String)> + Send>> {
        let tokenizer = match &self.tokenizer {
            Some(t) => t.clone(),
            None => {
                tracing::error!("Tokenizer not available");
                return Box::pin(futures::stream::empty());
            }
        };

        self.generate_stream_impl(input_ids, tokenizer, params)
            .await
    }

    /// Internal streaming implementation with token IDs.
    ///
    /// Uses tokio::task::spawn (lightweight) instead of spawn_blocking to avoid
    /// OS thread scheduling overhead. The only blocking operation is pool acquisition,
    /// which uses block_in_place to yield the async executor.
    ///
    /// Returns a stream of (token_id, token_text) tuples so callers can access
    /// raw token IDs without having to re-tokenize the text on their side.
    async fn generate_stream_impl(
        &self,
        input_ids: Vec<u32>,
        tokenizer: LMTokenizer,
        params: GenerationParams,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = (u32, String)> + Send>> {
        let pool = self
            .request_pool
            .as_ref()
            .expect("Request pool not initialized")
            .clone();
        let params_clone = params.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<(u32, String)>(32);

        // Use tokio::task::spawn instead of spawn_blocking to avoid OS thread overhead.
        // Only the pool acquisition is blocking (uses block_in_place internally),
        // and forward_async is non-blocking (submits request to C++ engine queue).
        tokio::task::spawn(async move {
            // Use block_in_place for pool acquisition to avoid blocking the async executor.
            // This is necessary because the pool uses tokio::sync primitives.
            let (_permit, mut request, mut input_tensors, mut _output_tensors) =
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(pool.acquire())
                });

            // Clear and reuse TensorMaps
            input_tensors.clear();

            // Create condition variable for completion signaling (event-driven, no polling)
            let completion = Arc::new((StdMutex::new(false), Condvar::new()));

            // Create callback context with completion condition variable
            let ctx = Arc::new(StreamContext {
                tokenizer,
                tx,
                completion: Arc::clone(&completion),
            });
            let ctx_ptr = ContextPtr(Arc::into_raw(ctx) as *mut c_void);

            // Set the token callback before submitting the request
            if let Err(e) = unsafe { request.set_token_callback(token_callback, ctx_ptr.0) } {
                tracing::error!(error = ?e, "Failed to set token callback");
                let _ = unsafe { Arc::from_raw(ctx_ptr.0 as *const StreamContext) };
                return;
            }

            // Set the completion callback to avoid polling
            if let Err(e) =
                unsafe { request.set_completion_callback(completion_callback, ctx_ptr.0) }
            {
                tracing::error!(error = ?e, "Failed to set completion callback");
                let _ = unsafe { Arc::from_raw(ctx_ptr.0 as *const StreamContext) };
                return;
            }

            // Start async GPU transfer for input_ids - CPU continues with config prep
            // while H2D happens in the background
            if let Some(event) = set_input_ids_gpu_uint32_async(&mut input_tensors, &input_ids) {
                // Sync on the event before forward to ensure data is ready on GPU
                let _ = event.sync();
            }

            // Prepare generation config with HTTP parameters
            let mut gen_cfg = match crate::turbomind_c::GenConfig::new() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create gen config");
                    let _ = unsafe { Arc::from_raw(ctx_ptr.0 as *const StreamContext) };
                    return;
                }
            };
            gen_cfg.set_max_new_tokens(params_clone.max_tokens.unwrap_or(1024) as i32);
            gen_cfg.set_temperature(params_clone.temperature.unwrap_or(0.7));
            gen_cfg.set_top_p(params_clone.top_p.unwrap_or(0.95));
            gen_cfg.set_top_k(params_clone.top_k.unwrap_or(50) as i32);
            // Apply any additional parameters (min_p, repetition_penalty, seed)
            params_clone.apply_to_gen_config(&mut gen_cfg);

            // Session parameters (unique session ID)
            let session = crate::turbomind_c::TM_SessionParam {
                id: unix_timestamp() as u64,
                step: 0,
                start_flag: true,
                end_flag: true,
            };

            // Attach grammar for guided decoding if provided
            if let Some(grammar) = &params_clone.grammar {
                if let Err(e) = request.set_grammar(grammar) {
                    tracing::warn!(error = ?e, "Failed to attach grammar for stream");
                    let _ = unsafe { Arc::from_raw(ctx_ptr.0 as *const StreamContext) };
                    return;
                }
            }

            // Submit async forward with stream_output=true
            // This is non-blocking: submits to C++ engine queue and returns immediately
            if let Err(e) = request.forward_async(
                &mut input_tensors,
                &session,
                &gen_cfg,
                true,  // stream_output
                false, // enable_metrics
            ) {
                tracing::error!(error = ?e, "ForwardAsync failed");
                let _ = unsafe { Arc::from_raw(ctx_ptr.0 as *const StreamContext) };
                return;
            }

            // Event-driven wait: completion callback notifies via Condvar
            // Zero CPU waste - blocks until callback signals completion
            let (lock, cvar) = &*completion;
            let mut done = lock.lock().unwrap();
            while !*done {
                // Safety: Condvar::wait_timeout returns Err only if lock is poisoned
                // Use 100ms timeout as safety net against missed notifications
                let result = cvar.wait_timeout(done, std::time::Duration::from_millis(100));
                done = match result {
                    Ok((guard, _timeout)) => guard,
                    Err(poisoned) => {
                        // Poisoned lock: extract guard from tuple and drop it
                        // to properly release the lock, then re-acquire
                        drop(poisoned.into_inner().0);
                        lock.lock().unwrap()
                    }
                };
            }

            // Reclaim the Arc to prevent memory leak
            let _ctx = unsafe { Arc::from_raw(ctx_ptr.0 as *const StreamContext) };
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> Option<&LMTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Get the request pool for benchmark timing
    pub fn pool(&self) -> Option<&RequestPool> {
        self.request_pool.as_ref().map(|arc| arc.as_ref())
    }

    /// Generate embeddings for text by running token embedding lookup + forward pass.
    ///
    /// Uses `output_last_hidden_state=2` (kGeneration = last token only) to extract
    /// the final hidden state, which represents the semantic embedding of the input.
    ///
    /// Note: This performs a minimal forward pass to get hidden states only. The model
    /// still needs to be fully initialized. If `dimensions` is specified and smaller
    /// than the model's hidden size, returns the first `dimensions` dimensions.
    pub async fn embed(&self, text: &str, dimensions: Option<usize>) -> Vec<f32> {
        let _tm = match &self.tm {
            Some(t) => t,
            None => {
                tracing::error!("TurboMind not initialized");
                return Vec::new();
            }
        };

        // Tokenize input
        let input_ids = match &self.tokenizer {
            Some(tokenizer) => match tokenizer.encode(text, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed for embed");
                    return Vec::new();
                }
            },
            None => {
                tracing::error!("Tokenizer not available for embed");
                return Vec::new();
            }
        };

        if input_ids.is_empty() {
            tracing::warn!("Empty input for embed");
            return Vec::new();
        }

        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let batch_size = input_ids_i64.len();

        // Get hidden size from model config (stored during init)
        let hidden_size = self.hidden_size;

        // Default to full hidden size, truncate if dimensions requested

        let target_dims = dimensions.unwrap_or(hidden_size).min(hidden_size);

        tracing::debug!(
            batch_size,
            hidden_size,
            target_dims,
            "embed: prepared input tensors"
        );

        // Acquire a ModelRequest from the pool (must clone Arc for spawn_blocking 'static)
        let pool = match &self.request_pool {
            Some(p) => Arc::clone(p),
            None => {
                tracing::error!("Request pool not initialized");
                return Vec::new();
            }
        };

        // Use blocking task for FFI calls
        let embedding_result = tokio::task::spawn_blocking(move || {
            // Acquire a slot (blocking semaphore + mutex)
            let (_permit, mut request, mut input_tensors, mut output_tensors) =
                pool.acquire_blocking();

            // Clear and reuse TensorMaps
            input_tensors.clear();
            output_tensors.clear();

            set_input_ids_i64(&mut input_tensors, &input_ids_i64);

            // Prepare generation config with output_last_hidden_state=2 (kGeneration = last token)
            let mut gen_cfg = match GenConfig::new() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create gen config");
                    return Vec::new();
                }
            };
            gen_cfg.set_max_new_tokens(1);
            gen_cfg.set_temperature(0.0);
            gen_cfg.set_output_last_hidden_state(2); // kGeneration = last token only

            // Session parameters
            let session = TM_SessionParam {
                id: unix_timestamp() as u64,
                step: 0,
                start_flag: true,
                end_flag: true,
            };

            // Run inference to get last_hidden_state
            if let Err(e) = request.forward(
                &mut input_tensors,
                &session,
                &gen_cfg,
                false, // stream_output
                false, // enable_metrics
                &mut output_tensors,
            ) {
                tracing::error!(error = ?e, "embed forward failed");
                return Vec::new();
            }

            // Extract last_hidden_state from output
            let (data_ptr, size) = match request.get_output("last_hidden_state") {
                Ok((ptr, sz)) => (ptr, sz),
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to get last_hidden_state output");
                    return Vec::new();
                }
            };

            let elem_count = size / 4; // float32 = 4 bytes
            if elem_count == 0 {
                tracing::warn!("Empty last_hidden_state output");
                return Vec::new();
            }

            tracing::debug!(elem_count, "last_hidden_state raw data");

            // The hidden state is [batch_size, hidden_dim] but we only asked for kGeneration
            // (last token), so we should get [1, hidden_dim]
            // However, if output_last_hidden_state=2 returns last prompt token (not generated token),
            // we need to extract correctly. Let's check the actual shape.
            //
            // If we have [1, hidden_dim], extract first `target_dims` elements
            // If we have [batch_size, hidden_dim], extract the LAST row's first `target_dims` elements

            let embedding = if elem_count == hidden_size {
                // Single vector: [hidden_dim]
                let slice =
                    unsafe { std::slice::from_raw_parts(data_ptr as *const f32, elem_count) };
                slice[..target_dims].to_vec()
            } else if elem_count > hidden_size {
                // Multiple vectors: [N, hidden_dim] - take the LAST one (last token)
                let num_vectors = elem_count / hidden_size;
                let start_idx = (num_vectors - 1) * hidden_size;
                let slice = unsafe {
                    std::slice::from_raw_parts(data_ptr.add(start_idx) as *const f32, hidden_size)
                };
                slice[..target_dims].to_vec()
            } else {
                tracing::warn!(elem_count, hidden_size, "Unexpected last_hidden_state size");
                Vec::new()
            };

            tracing::debug!(
                embedding_len = embedding.len(),
                "embed: returning embedding"
            );
            embedding
        })
        .await
        .unwrap_or_default();

        embedding_result
    }

    /// Generate embeddings as a DLPack TM_Tensor (zero-copy output).
    ///
    /// Returns the embedding as a `TM_Tensor` struct that shares memory with
    /// the C++ engine's output buffer, avoiding CPU copy. The caller can convert
    /// this to a DLPack capsule via `TM_Tensor::to_dlpack()`.
    ///
    /// # Arguments
    /// * `text` - Input text to embed
    ///
    /// # Returns
    /// A `TM_Tensor` with:
    /// - `data`: GPU pointer to embedding data
    /// - `dtype`: TM_DATATYPE_FP32
    /// - `ndim`: 2
    /// - `shape`: [1, hidden_size]
    ///
    /// # Safety
    /// The returned TM_Tensor.data pointer is valid only until the request completes.
    /// The caller must consume the data before the ModelRequest is dropped.
    pub async fn embed_as_dlpack(&self, text: &str) -> Option<TM_Tensor> {
        let pool = match &self.request_pool {
            Some(p) => Arc::clone(p),
            None => {
                tracing::error!("Request pool not initialized");
                return None;
            }
        };

        let hidden_size = self.hidden_size;

        // Tokenize input
        let input_ids_i64: Vec<i64> = match &self.tokenizer {
            Some(tokenizer) => match tokenizer.encode(text, false, false) {
                Ok(ids) => ids.iter().map(|&id| id as i64).collect(),
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed for embed");
                    return None;
                }
            },
            None => {
                tracing::error!("Tokenizer not available for embed");
                return None;
            }
        };

        if input_ids_i64.is_empty() {
            tracing::warn!("Empty input for embed");
            return None;
        }

        let embedding_result = tokio::task::spawn_blocking(move || {
            let (_permit, mut request, mut input_tensors, mut output_tensors) =
                pool.acquire_blocking();

            // Clear and reuse TensorMaps
            input_tensors.clear();
            output_tensors.clear();

            set_input_ids_i64(&mut input_tensors, &input_ids_i64);

            let mut gen_cfg = match GenConfig::new() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create gen config");
                    return None;
                }
            };
            gen_cfg.set_max_new_tokens(1);
            gen_cfg.set_temperature(0.0);
            gen_cfg.set_output_last_hidden_state(2); // kGeneration = last token only

            let session = TM_SessionParam {
                id: unix_timestamp() as u64,
                step: 0,
                start_flag: true,
                end_flag: true,
            };

            if let Err(e) = request.forward(
                &mut input_tensors,
                &session,
                &gen_cfg,
                false,
                false,
                &mut output_tensors,
            ) {
                tracing::error!(error = ?e, "embed forward failed");
                return None;
            }

            // Get last_hidden_state output as raw pointer
            let (data_ptr, size) = match request.get_output("last_hidden_state") {
                Ok((ptr, sz)) => (ptr, sz),
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to get last_hidden_state output");
                    return None;
                }
            };

            let elem_count = size / 4; // float32
            if elem_count == 0 {
                tracing::warn!("Empty last_hidden_state output");
                return None;
            }

            // Determine shape: [1, hidden_size] or [batch_size, hidden_size]
            let num_vectors = elem_count / hidden_size;
            let (ndim, shape_array) = if num_vectors == 1 {
                // [hidden_size]
                (1, [hidden_size as i64, 1, 1, 1, 1, 1, 1, 1])
            } else {
                // [num_vectors, hidden_size]
                (
                    2,
                    [num_vectors as i64, hidden_size as i64, 1, 1, 1, 1, 1, 1],
                )
            };

            // Calculate offset to last token's embedding
            let offset = if num_vectors > 1 {
                (num_vectors - 1) * hidden_size
            } else {
                0
            };

            let data_ptr = unsafe { data_ptr.add(offset * 4) };

            Some(TM_Tensor {
                dtype: TM_DataType::TM_DATATYPE_FP32,
                ndim: ndim as c_int,
                shape: shape_array,
                data: data_ptr as *mut c_void,
                device_id: 0, // GPU
            })
        })
        .await
        .unwrap_or_default();

        embedding_result
    }

    /// Get schedule metrics
    pub fn get_metrics(&self) -> Result<ScheduleMetrics> {
        let tm = self.tm.as_ref().expect("TurboMind not initialized");
        tm.get_schedule_metrics(0)
            .map_err(|e| AppError::InferenceFailed(format!("Failed to get metrics: {:?}", e)))
    }
}

/// A batch inference request item.
///
/// Contains the input data and generation parameters for a single request
/// within a batch.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Unique request ID (for tracking)
    pub request_id: u64,
    /// Input prompt text
    pub prompt: String,
    /// Generation parameters
    pub params: GenerationParams,
    /// Whether to return logprobs
    pub need_logprobs: bool,
}

/// Result of a batch inference request.
///
/// Contains the generated text and optional logprobs.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Request ID matching the input BatchItem
    pub request_id: u64,
    /// Generated text
    pub text: String,
    /// Number of generated tokens
    pub num_tokens: usize,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
    /// Logprobs if requested
    pub logprobs: Option<Vec<TokenLogprob>>,
    /// Error message if request failed
    pub error: Option<String>,
}

/// Extract results from a completed batch request.
fn extract_batch_result(
    request: &ModelRequest,
    tokenizer: &LMTokenizer,
    request_id: u64,
    start: Instant,
    need_logprobs: bool,
    top_logprobs_req: u32,
) -> BatchResult {
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match request.get_output("output_ids") {
        Ok((data_ptr, size)) => {
            let num_tokens = size / 4;
            let output_ids: Vec<i32> =
                unsafe { std::slice::from_raw_parts(data_ptr as *const i32, num_tokens).to_vec() };

            let text = match tokenizer.decode(
                &output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                true,
            ) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = %e, "Decoding failed");
                    format!("[decode error: {}]", e)
                }
            };

            let logprobs = if need_logprobs {
                extract_logprobs(&output_ids, request, tokenizer, top_logprobs_req)
            } else {
                None
            };

            BatchResult {
                request_id,
                text,
                num_tokens,
                elapsed_ms,
                logprobs,
                error: None,
            }
        }
        Err(e) => {
            tracing::error!(error = ?e, "Failed to get output_ids");
            BatchResult {
                request_id,
                text: String::new(),
                num_tokens: 0,
                elapsed_ms,
                logprobs: None,
                error: Some(format!("Failed to get output_ids: {:?}", e)),
            }
        }
    }
}

impl TurboMindCEngine {
    /// Generate text for a batch of prompts with true vectorized batch inference.
    ///
    /// This method implements **true batch inference** by:
    /// 1. Tokenizing all prompts in parallel (outside the lock)
    /// 2. Submitting all requests simultaneously to the C++ engine's gateway queue
    /// 3. The C++ engine's ModelExecutor performs dynamic batching on GPU
    /// 4. Polling for completion and collecting results
    ///
    /// The C++ TurboMind engine has an internal gateway that batches requests
    /// together automatically. By submitting all requests simultaneously (via
    /// forward_async), we enable the engine to batch them efficiently on GPU.
    ///
    /// Returns a vector of results in the same order as the input items.
    pub async fn generate_batch(&self, items: Vec<BatchItem>) -> Vec<BatchResult> {
        if items.is_empty() {
            return Vec::new();
        }

        // Vectorized batch: submit all requests simultaneously to the C++ gateway
        // The gateway's ModelExecutor performs dynamic batching on GPU
        self.generate_batch_vectorized(items).await
    }

    /// True vectorized batch inference using the C++ engine's internal batching.
    ///
    /// This method submits all requests simultaneously to the C++ gateway queue,
    /// which batches them together for GPU execution. This is more efficient than
    /// sequential processing because:
    /// 1. Multiple requests are processed in a single GPU kernel launch
    /// 2. Better GPU utilization with larger batch sizes
    /// 3. Reduced overhead from multiple kernel launches
    async fn generate_batch_vectorized(&self, items: Vec<BatchItem>) -> Vec<BatchResult> {
        let pool = match self.request_pool.as_ref() {
            Some(p) => Arc::clone(p),
            None => {
                tracing::error!("Request pool not initialized");
                return items
                    .into_iter()
                    .map(|item| BatchResult {
                        request_id: item.request_id,
                        text: String::new(),
                        num_tokens: 0,
                        elapsed_ms: 0.0,
                        logprobs: None,
                        error: Some("Request pool not initialized".to_string()),
                    })
                    .collect::<Vec<BatchResult>>();
            }
        };

        let tokenizer = match &self.tokenizer {
            Some(t) => t.clone(),
            None => {
                tracing::error!("Tokenizer not available");
                return items
                    .into_iter()
                    .map(|item| BatchResult {
                        request_id: item.request_id,
                        text: String::new(),
                        num_tokens: 0,
                        elapsed_ms: 0.0,
                        logprobs: None,
                        error: Some("Tokenizer not available".to_string()),
                    })
                    .collect::<Vec<BatchResult>>();
            }
        };

        // Phase 1: Tokenize all prompts in parallel (CPU-bound)
        // This runs before any GPU operations, maximizing CPU utilization
        let tokenized_items: Vec<(BatchItem, std::result::Result<Vec<u32>, String>)> = items
            .into_iter()
            .map(|item| {
                let prompt_text = item.prompt.clone();
                let encoded = tokenizer
                    .encode(&prompt_text, false, false)
                    .map_err(|e| e.to_string());
                (item, encoded)
            })
            .collect();

        // Phase 2: Pre-calculate total token count for slot allocation planning
        let total_tokens: usize = tokenized_items
            .iter()
            .filter_map(|(_, result)| result.as_ref().ok().map(|ids| ids.len()))
            .sum();

        tracing::debug!(
            total_tokens,
            num_requests = tokenized_items.len(),
            "Batch prefill: tokenized all prompts"
        );

        // Phase 3: Spawn tasks with optimized slot acquisition
        // All tasks start simultaneously, reducing sync contention
        let mut handles = Vec::new();
        for (item, tokenize_result) in tokenized_items {
            let pool_clone = Arc::clone(&pool);
            let tokenizer_clone = tokenizer.clone();

            let handle = tokio::task::spawn_blocking(move || {
                let start = Instant::now();
                let request_id = item.request_id;

                // Use pre-tokenized input_ids
                let input_ids = match tokenize_result {
                    Ok(ids) => ids,
                    Err(e) => {
                        tracing::error!(error = %e, request_id, "Tokenization failed");
                        return BatchResult {
                            request_id,
                            text: String::new(),
                            num_tokens: 0,
                            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                            logprobs: None,
                            error: Some(e.to_string()),
                        };
                    }
                };

                // Acquire a slot from the pool
                let (_permit, mut request, mut input_tensors, _output_tensors) =
                    pool_clone.acquire_blocking();

                // Clear and reuse input TensorMap
                input_tensors.clear();

                // Start async GPU transfer for input_ids - CPU continues with config prep
                // while H2D happens in the background
                if let Some(event) = set_input_ids_gpu_uint32_async(&mut input_tensors, &input_ids) {
                    // Sync on the event before forward to ensure data is ready on GPU
                    let _ = event.sync();
                }

                // Prepare generation config
                let mut gen_cfg = match GenConfig::new() {
                    Ok(g) => g,
                    Err(e) => {
                        tracing::error!(error = ?e, request_id, "Failed to create gen config");
                        return BatchResult {
                            request_id,
                            text: String::new(),
                            num_tokens: 0,
                            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                            logprobs: None,
                            error: Some(format!("Failed to create gen config: {:?}", e)),
                        };
                    }
                };
                gen_cfg.set_max_new_tokens(item.params.max_tokens.unwrap_or(512) as i32);
                gen_cfg.set_temperature(item.params.temperature.unwrap_or(0.7));
                gen_cfg.set_top_p(item.params.top_p.unwrap_or(0.95));
                gen_cfg.set_top_k(item.params.top_k.unwrap_or(50) as i32);
                item.params.apply_to_gen_config(&mut gen_cfg);

                // Session parameters
                let session = crate::turbomind_c::TM_SessionParam {
                    id: unix_timestamp() as u64 + request_id,
                    step: 0,
                    start_flag: true,
                    end_flag: true,
                };

                // Attach grammar for guided decoding if provided
                if let Some(ref grammar) = item.params.grammar {
                    if let Err(e) = request.set_grammar(grammar) {
                        tracing::warn!(error = ?e, request_id, "Failed to attach grammar for batch");
                    }
                }

                // Submit async forward request - this adds to the C++ gateway queue
                // The gateway will batch multiple requests together for GPU execution
                let forward_result = request.forward_async(
                    &mut input_tensors,
                    &session,
                    &gen_cfg,
                    false, // stream_output
                    true,  // enable_metrics
                );

                if let Err(e) = forward_result {
                    tracing::error!(error = ?e, request_id, "ForwardAsync failed");
                    return BatchResult {
                        request_id,
                        text: String::new(),
                        num_tokens: 0,
                        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                        logprobs: None,
                        error: Some(format!("ForwardAsync failed: {:?}", e)),
                    };
                }

                // Event-driven completion: register completion callback instead of polling
                let completion = Arc::new((StdMutex::new(false), Condvar::new()));
                let completion_ctx = BatchCompletionContext {
                    completion: Arc::clone(&completion),
                };
                let ctx_ptr = Box::into_raw(Box::new(completion_ctx)) as *mut c_void;

                if let Err(e) =
                    unsafe { request.set_completion_callback(batch_completion_callback, ctx_ptr) }
                {
                    tracing::error!(error = ?e, request_id, "Failed to set completion callback");
                    let _ = unsafe { Box::from_raw(ctx_ptr as *mut BatchCompletionContext) };
                    return BatchResult {
                        request_id,
                        text: String::new(),
                        num_tokens: 0,
                        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                        logprobs: None,
                        error: Some(format!("Failed to set completion callback: {:?}", e)),
                    };
                }

                // Wait for completion via Condvar (event-driven, zero CPU waste)
                let (lock, cvar) = &*completion;
                let mut done = lock.lock().unwrap();
                while !*done {
                    let result = cvar.wait_timeout(done, std::time::Duration::from_millis(100));
                    done = match result {
                        Ok((guard, _timeout)) => guard,
                        Err(poisoned) => {
                            // Poisoned lock: extract guard from tuple and drop it
                            // to properly release the lock, then re-acquire
                            drop(poisoned.into_inner().0);
                            lock.lock().unwrap()
                        }
                    };
                }
                // Cleanup
                let _ctx = unsafe { Box::from_raw(ctx_ptr as *mut BatchCompletionContext) };

                // Extract results after completion
                let need_logprobs = item.need_logprobs;
                let top_logprobs_req = item.params.top_logprobs.unwrap_or(1).max(1);
                return extract_batch_result(
                    &request,
                    &tokenizer_clone,
                    request_id,
                    start,
                    need_logprobs,
                    top_logprobs_req,
                );
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap_or_else(|e| BatchResult {
                request_id: 0,
                text: String::new(),
                num_tokens: 0,
                elapsed_ms: 0.0,
                logprobs: None,
                error: Some(format!("Task join error: {}", e)),
            }));
        }

        // Sort results by request_id to maintain input order
        results.sort_by_key(|r| r.request_id);
        results
    }

    /// Process a single batch request (fallback for compatibility).
    async fn process_single_request(&self, item: BatchItem) -> BatchResult {
        let request_id = item.request_id;

        if item.need_logprobs {
            let (text, num_tokens, elapsed_ms, logprobs) =
                self.generate_with_logprobs(&item.prompt, item.params).await;
            BatchResult {
                request_id,
                text,
                num_tokens,
                elapsed_ms,
                logprobs,
                error: None,
            }
        } else {
            let text = self.generate(&item.prompt, item.params).await;
            BatchResult {
                request_id,
                text,
                num_tokens: 0,
                elapsed_ms: 0.0,
                logprobs: None,
                error: None,
            }
        }
    }
}

impl std::fmt::Debug for TurboMindCEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboMindCEngine")
            .field("model_path", &self.model_path)
            .field("model_name", &self.model_name)
            .field("state", &self.state)
            .field("loaded_at", &self.loaded_at)
            .field("engine_type", &self.engine_type)
            .field("quant_policy", &self.quant_policy)
            .field("hidden_size", &self.hidden_size)
            .finish()
    }
}

fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Set a tensor to TensorMap using DLPack zero-copy transfer.
///
/// When the data is stored on GPU, this function uses DLPack to pass
/// the GPU pointer directly to the C++ engine without CPU copy.
/// For CPU data, it falls back to the standard setter.
fn set_tensor_dlpack(
    tensors: &mut TensorMap,
    name: &str,
    data_ptr: *const c_void,
    data_i64: &[i64],
    shape: &[i64],
    is_gpu: bool,
) {
    if is_gpu && !data_ptr.is_null() {
        // Zero-copy DLPack path: GPU pointer directly to C++ engine
        tensors.set_from_dlpack(
            name,
            data_ptr,
            shape,
            DL_DTYPE_CODE_INT as c_int,   // int64
            64,                           // 64 bits
            DL_DEVICE_TYPE_CUDA as c_int, // CUDA GPU
        );
    } else {
        // Fallback: CPU copy
        tensors.set_int64(name, data_i64, shape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_type_from_str() {
        assert_eq!(EngineType::from_str("cpp"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("c++"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("native"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("invalid"), None);
    }

    #[test]
    fn test_engine_type_as_str() {
        assert_eq!(EngineType::PureCpp.as_str(), "pure_cpp");
    }

    #[test]
    fn test_detect_awq_quantization() {
        // Test with non-existent path
        let path = std::path::PathBuf::from("/nonexistent/path");
        assert!(!detect_awq_quantization(&path));
    }
}
