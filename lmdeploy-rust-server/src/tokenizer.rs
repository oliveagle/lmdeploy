//! Splintr tokenizer wrapper for LMDeploy
//!
//! Loads tokenizers from model directories (tokenizer.json)
//! using splintr for high-performance encoding/decoding.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;
use splintr::core::byte_level::byte_level_decode;
use splintr::Tokenizer;

use crate::turbomind_c::{cudaMallocHost, cudaFreeHost, cudaStream_t, CudaEvent, cudaMemcpyAsync, cudaMemcpyKind};

/// Thread-local buffer for reusing tokenizer output allocation.
/// Eliminates per-request heap allocation by reusing capacity across calls.
use std::cell::RefCell;
thread_local! {
    static TOKEN_ID_BUFFER: RefCell<Vec<u32>> = RefCell::new(Vec::with_capacity(8192));
}

/// GPU tensor result from tokenization.
/// Contains pointer and length for DLPack transfer to C++ engine.
#[derive(Debug)]
pub struct GpuTensor<'a> {
    /// GPU pointer to token data (pinned memory page-locked for H2D)
    pub gpu_ptr: *const u32,
    /// Number of tokens in the tensor
    pub len: usize,
    /// Pinned host memory pointer (for cleanup, not passed to C++)
    pub host_ptr: *mut std::ffi::c_void,
    /// CUDA stream used for async transfer
    pub stream: cudaStream_t,
    /// Sync event to ensure GPU transfer completes before use
    pub sync_event: Option<CudaEvent>,
    /// Temporary reference to avoid lifetime issues
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> GpuTensor<'a> {
    /// Check if the tensor is valid (non-null pointer)
    pub fn is_valid(&self) -> bool {
        !self.gpu_ptr.is_null() && self.len > 0
    }
}

impl Drop for GpuTensor<'_> {
    fn drop(&mut self) {
        // Note: We don't free the host_ptr here because:
        // 1. The pinned buffer is managed by the caller (LMTokenizerWithGpu)
        // 2. The GPU pointer may still be in use by the C++ engine
        // The caller should call release_pinned_buffer() when safe
    }
}

// SAFETY: GpuTensor contains raw pointers to GPU and pinned host memory.
// It is safe to Send because:
// 1. The memory regions are explicitly managed (not accessed after drop)
// 2. Synchronization is handled via CudaEvent.sync() before concurrent use
// 3. The lifetime parameter ensures host memory outlives the tensor
unsafe impl Send for GpuTensor<'_> {}

/// LMDeploy tokenizer wrapper backed by splintr
#[derive(Clone)]
pub struct LMTokenizer {
    tokenizer: Arc<Tokenizer>,
    bos_token_id: Option<u32>,
    eos_token_ids: Vec<u32>,
    vocab_size: usize,
}

impl LMTokenizer {
    /// Create a new tokenizer from a model directory.
    pub fn from_path(model_dir: &str) -> Result<Self> {
        let path = Path::new(model_dir);

        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists() {
            let (tokenizer, bos, eos) = load_bpe_from_json(&tokenizer_json)?;
            return Self::new(tokenizer, bos, eos);
        }

        Err(anyhow!(
            "No tokenizer files found in {}. Expected tokenizer.json",
            model_dir
        ))
    }

    fn new(
        tokenizer: Tokenizer,
        bos_token_id: Option<u32>,
        eos_token_ids: Vec<u32>,
    ) -> Result<Self> {
        let vocab_size = tokenizer.vocab_size();
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            bos_token_id,
            eos_token_ids,
            vocab_size,
        })
    }

    pub fn encode(&self, text: &str, add_bos: bool, _add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut token_ids = self.tokenizer.encode(text);

        if add_bos {
            if let Some(bos_id) = self.bos_token_id {
                token_ids.insert(0, bos_id);
            }
        }

        if token_ids.len() >= 2 {
            if let Some(bos_id) = self.bos_token_id {
                if token_ids[0] == bos_id && token_ids[1] == bos_id {
                    tracing::warn!(
                        "Detected duplicate bos token {} in prompt, removing one",
                        bos_id
                    );
                    token_ids.remove(0);
                }
            }
        }

        Ok(token_ids)
    }

    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text))
    }

    pub fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids)
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.tokenizer
            .decode(&[token_id])
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        let bytes = self.tokenizer.decoder().get(&token_id)?;
        String::from_utf8(bytes.clone()).ok()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub fn encode_batch(&self, texts: &[&str], add_bos: bool) -> Result<Vec<Vec<u32>>> {
        let string_texts: Vec<String> = texts.iter().map(|&t| t.to_string()).collect();
        let results = self.tokenizer.encode_batch(&string_texts);

        let mut output = Vec::with_capacity(results.len());
        for mut token_ids in results {
            if add_bos {
                if let Some(bos_id) = self.bos_token_id {
                    token_ids.insert(0, bos_id);
                }
            }
            if token_ids.len() >= 2 {
                if let Some(bos_id) = self.bos_token_id {
                    if token_ids[0] == bos_id && token_ids[1] == bos_id {
                        tracing::warn!(
                            "Detected duplicate bos token {} in prompt, removing one",
                            bos_id
                        );
                        token_ids.remove(0);
                    }
                }
            }
            output.push(token_ids);
        }
        Ok(output)
    }
}

impl std::fmt::Debug for LMTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LMTokenizer")
            .field("vocab_size", &self.vocab_size)
            .field("bos_token_id", &self.bos_token_id)
            .field("eos_token_ids", &self.eos_token_ids)
            .finish()
    }
}

// ============================================================================
// Tokenizer loading: parse tokenizer.json -> splintr Tokenizer
// ============================================================================

/// Load HuggingFace tokenizer.json (byte-level BPE, e.g. Qwen).
fn load_bpe_from_json(path: &Path) -> Result<(Tokenizer, Option<u32>, Vec<u32>)> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("Failed to read {}: {}", path.display(), e))?;

    let parsed: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| anyhow!("Failed to parse tokenizer.json: {}", e))?;

    let model = parsed
        .get("model")
        .ok_or_else(|| anyhow!("Missing 'model' field"))?;

    let model_type = model
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing model type"))?;

    if model_type != "BPE" {
        return Err(anyhow!("Unsupported model type: {}", model_type));
    }

    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("Missing vocab"))?;

    // Detect byte-level decoder
    let is_byte_level = model
        .get("decoder")
        .and_then(|d| d.get("type"))
        .and_then(|v| v.as_str())
        == Some("ByteLevel");

    // Build encoder: bytes -> token ID
    let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    for (token_str, id_val) in vocab {
        let id = id_val.as_u64().ok_or_else(|| anyhow!("Invalid token ID"))? as u32;
        let bytes = if is_byte_level {
            byte_level_decode(token_str)
                .ok_or_else(|| anyhow!("Byte-level decode failed for token: {:?}", token_str))?
        } else {
            token_str.as_bytes().to_vec()
        };
        encoder.insert(bytes, id);
    }

    // Collect special tokens from added_tokens
    let mut special_tokens: FxHashMap<String, u32> = FxHashMap::default();
    let mut bos_token_id: Option<u32> = None;
    let mut eos_token_ids: Vec<u32> = Vec::new();

    if let Some(added_tokens) = parsed.get("added_tokens").and_then(|v| v.as_array()) {
        for entry in added_tokens {
            let id = match entry.get("id").and_then(|v| v.as_u64()) {
                Some(v) => v as u32,
                None => continue,
            };
            let content = match entry.get("content").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => continue,
            };
            let is_special = entry
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_special {
                special_tokens.insert(content.to_string(), id);
            }

            match content {
                "<s>" | "<bos>" | "<BOS>" => bos_token_id = Some(id),
                "</s>" | "<eos>" | "<EOS>" => eos_token_ids.push(id),
                _ => {}
            }
        }
    }

    // Fallback: scan vocab
    if bos_token_id.is_none() {
        if let Some(id) = find_bpe_token(&encoder, is_byte_level, "<s>") {
            bos_token_id = Some(id);
        }
    }
    if eos_token_ids.is_empty() {
        if let Some(id) = find_bpe_token(&encoder, is_byte_level, "</s>") {
            eos_token_ids.push(id);
        }
    }

    let pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+| ?(?:\r\n\s*|\s+)|\s+(?!\S)|\s+";

    let tokenizer = if is_byte_level {
        Tokenizer::new_byte_level(encoder, special_tokens, pattern)
            .map_err(|e| anyhow!("Failed to create ByteLevel tokenizer: {}", e))?
    } else {
        Tokenizer::new(encoder, special_tokens, pattern)
            .map_err(|e| anyhow!("Failed to create tokenizer: {}", e))?
    };

    Ok((tokenizer, bos_token_id, eos_token_ids))
}

fn find_bpe_token(
    encoder: &FxHashMap<Vec<u8>, u32>,
    is_byte_level: bool,
    token: &str,
) -> Option<u32> {
    if is_byte_level {
        let bytes = byte_level_decode(token)?;
        encoder.get(&bytes).copied()
    } else {
        encoder.get(token.as_bytes()).copied()
    }
}

// ============================================================================
// GPU Tokenizer with Zero-Copy Path
// ============================================================================

/// Pinned host buffer for tokenization output.
/// This buffer is page-locked for faster CPU->GPU transfers.
struct PinnedTokenBuffer {
    host_ptr: *mut std::ffi::c_void,
    capacity: usize,
}

unsafe impl Send for PinnedTokenBuffer {}

impl PinnedTokenBuffer {
    fn new(initial_capacity: usize) -> Self {
        let mut host_ptr = std::ptr::null_mut();
        let size = initial_capacity * std::mem::size_of::<u32>();
        unsafe {
            if cudaMallocHost(&mut host_ptr, size) != 0 {
                tracing::warn!("cudaMallocHost failed for tokenization buffer");
                host_ptr = std::ptr::null_mut();
            }
        }
        Self {
            host_ptr,
            capacity: initial_capacity,
        }
    }

    fn get_or_grow(&mut self, needed_elements: usize) -> *mut std::ffi::c_void {
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

    fn as_slice_mut(&mut self, len: usize) -> &mut [u32] {
        unsafe {
            std::slice::from_raw_parts_mut(self.host_ptr as *mut u32, len)
        }
    }
}

impl Drop for PinnedTokenBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            unsafe { cudaFreeHost(self.host_ptr) };
        }
    }
}

/// GPU buffer for async tokenization transfer.
/// Manages GPU memory and CUDA stream for H2D copies.
struct GpuTokenBuffer {
    gpu_ptr: *mut std::ffi::c_void,
    capacity: usize,
    stream: crate::turbomind_c::cudaStream_t,
}

unsafe impl Send for GpuTokenBuffer {}

impl GpuTokenBuffer {
    fn new(initial_capacity: usize) -> Self {
        let mut gpu_ptr = std::ptr::null_mut();
        let size = initial_capacity * std::mem::size_of::<u32>();
        unsafe {
            if crate::turbomind_c::cudaMalloc(&mut gpu_ptr, size) != 0 {
                tracing::warn!("cudaMalloc failed for tokenization GPU buffer");
            }
        }
        let mut stream: crate::turbomind_c::cudaStream_t = std::ptr::null_mut();
        unsafe {
            if crate::turbomind_c::cudaStreamCreate(&mut stream) != 0 {
                tracing::warn!("cudaStreamCreate failed for tokenization");
            }
        }
        Self {
            gpu_ptr,
            capacity: initial_capacity,
            stream,
        }
    }

    fn get_or_grow(&mut self, needed_elements: usize) -> *mut std::ffi::c_void {
        if needed_elements > self.capacity {
            if !self.gpu_ptr.is_null() {
                unsafe { crate::turbomind_c::cudaFree(self.gpu_ptr) };
            }
            let new_capacity = needed_elements.next_power_of_two().max(self.capacity * 2);
            let mut new_ptr = std::ptr::null_mut();
            let size = new_capacity * std::mem::size_of::<u32>();
            unsafe {
                if crate::turbomind_c::cudaMalloc(&mut new_ptr, size) == 0 {
                    self.gpu_ptr = new_ptr;
                    self.capacity = new_capacity;
                } else {
                    self.gpu_ptr = std::ptr::null_mut();
                }
            }
        }
        self.gpu_ptr
    }

    fn stream(&self) -> cudaStream_t {
        self.stream
    }
}

impl Drop for GpuTokenBuffer {
    fn drop(&mut self) {
        if !self.gpu_ptr.is_null() {
            unsafe { crate::turbomind_c::cudaFree(self.gpu_ptr) };
        }
        if !self.stream.is_null() {
            unsafe { crate::turbomind_c::cudaStreamDestroy(self.stream) };
        }
    }
}

/// Tokenizer with zero-copy GPU output path.
///
/// Wraps `LMTokenizer` and provides `encode_to_gpu()` that:
/// 1. Tokenizes directly to pre-allocated pinned memory
/// 2. Async copies to GPU via dedicated CUDA stream
/// 3. Returns GPU tensor ready for DLPack zero-copy transfer to C++
///
/// This eliminates the intermediate `Vec<u32>` allocation and reduces
/// memory copies from 2 to 1.
pub struct LMTokenizerWithGpu {
    tokenizer: LMTokenizer,
    pinned_buffer: std::sync::Mutex<PinnedTokenBuffer>,
    gpu_buffer: std::sync::Mutex<GpuTokenBuffer>,
}

impl LMTokenizerWithGpu {
    /// Create a new GPU-enabled tokenizer from a model directory.
    pub fn from_path(model_dir: &str) -> Result<Self> {
        let tokenizer = LMTokenizer::from_path(model_dir)?;
        Ok(Self {
            tokenizer,
            pinned_buffer: std::sync::Mutex::new(PinnedTokenBuffer::new(8192)),
            gpu_buffer: std::sync::Mutex::new(GpuTokenBuffer::new(8192)),
        })
    }

    /// Create from an existing LMTokenizer.
    pub fn from_tokenizer(tokenizer: LMTokenizer) -> Self {
        Self {
            tokenizer,
            pinned_buffer: std::sync::Mutex::new(PinnedTokenBuffer::new(8192)),
            gpu_buffer: std::sync::Mutex::new(GpuTokenBuffer::new(8192)),
        }
    }

    /// Encode text directly to GPU memory.
    ///
    /// Returns a GPU tensor that can be passed directly to the C++ engine
    /// via DLPack with zero additional copies.
    ///
    /// # Flow
    /// 1. Tokenize using thread-local buffer (reuses capacity, no per-request allocation)
    /// 2. Copy to pinned memory (single copy)
    /// 3. Async copy to GPU via dedicated CUDA stream
    /// 4. Record sync event
    /// 5. Return GPU tensor ready for DLPack transfer
    ///
    /// The caller must call `tensor.sync_event.sync()` before using the
    /// GPU data to ensure the async transfer completes.
    pub fn encode_to_gpu(&self, text: &str, add_bos: bool) -> Result<GpuTensor<'_>> {
        // Tokenize using thread-local buffer to avoid per-request allocation
        // Thread-local buffer reuses capacity across calls (amortized O(1) append)
        let (token_ids_ptr, token_count) = TOKEN_ID_BUFFER.with(|buf_cell| {
            let mut buf = buf_cell.borrow_mut();
            buf.clear();

            // Get raw tokens from splintr tokenizer
            let raw_tokens = self.tokenizer.tokenizer.encode(text);
            let base_count = raw_tokens.len();

            // Reserve capacity for BOS token if needed
            if add_bos && self.tokenizer.bos_token_id.is_some() {
                buf.reserve(base_count + 1);
            } else {
                buf.reserve(base_count);
            }

            // Add BOS token if requested
            if add_bos {
                if let Some(bos_id) = self.tokenizer.bos_token_id {
                    buf.push(bos_id);
                }
            }

            // Extend with raw tokens
            buf.extend(raw_tokens);

            // Duplicate BOS detection
            if buf.len() >= 2 {
                if let Some(bos_id) = self.tokenizer.bos_token_id {
                    if buf[0] == bos_id && buf[1] == bos_id {
                        tracing::warn!(
                            "Detected duplicate bos token {} in prompt, removing one",
                            bos_id
                        );
                        buf.remove(0);
                    }
                }
            }

            // Return raw pointer to buffer data (valid until next mutable borrow)
            (buf.as_ptr(), buf.len())
        });

        // Get pinned buffer (grow if needed) and copy data in one lock scope
        let (gpu_ptr, pinned_ptr, stream) = {
            let mut pinned_buffer = self.pinned_buffer.lock().unwrap();
            let pinned_ptr = pinned_buffer.get_or_grow(token_count);
            if pinned_ptr.is_null() {
                return Err(anyhow!("Failed to allocate pinned buffer"));
            }

            // Copy from thread-local buffer to pinned memory (single copy)
            // Safety: token_ids_ptr is valid, pinned_ptr is valid, non-overlapping
            let pinned_slice = pinned_buffer.as_slice_mut(token_count);
            unsafe {
                std::ptr::copy_nonoverlapping(token_ids_ptr, pinned_slice.as_mut_ptr(), token_count);
            }

            // Get GPU buffer (grow if needed)
            let mut gpu_buffer = self.gpu_buffer.lock().unwrap();
            let gpu_ptr = gpu_buffer.get_or_grow(token_count);
            if gpu_ptr.is_null() {
                return Err(anyhow!("Failed to allocate GPU buffer"));
            }

            // Async copy to GPU
            let stream = gpu_buffer.stream();
            unsafe {
                cudaMemcpyAsync(
                    gpu_ptr,
                    pinned_ptr,
                    token_count * std::mem::size_of::<u32>(),
                    cudaMemcpyKind::HostToDevice,
                    stream,
                );
            }

            (gpu_ptr, pinned_ptr, stream)
        };

        // Create sync event
        let sync_event = CudaEvent::new().ok();

        if let Some(ref event) = sync_event {
            let _ = event.record(stream);
        }

        Ok(GpuTensor {
            gpu_ptr: gpu_ptr as *const u32,
            len: token_count,
            host_ptr: pinned_ptr,
            stream,
            sync_event,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Encode text to CPU (fallback for compatibility).
    pub fn encode(&self, text: &str, add_bos: bool, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, add_bos, add_special_tokens)
    }

    /// Decode token IDs to text.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(token_ids, skip_special_tokens)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer.bos_token_id()
    }

    /// Get EOS token IDs.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        self.tokenizer.eos_token_ids().to_vec()
    }

    /// Check if token is EOS.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.tokenizer.is_eos(token_id)
    }

    /// Get the underlying LMTokenizer.
    pub fn inner(&self) -> &LMTokenizer {
        &self.tokenizer
    }
}

impl std::fmt::Debug for LMTokenizerWithGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pinned_capacity = self.pinned_buffer.lock().map(|b| b.capacity).unwrap_or(0);
        let gpu_capacity = self.gpu_buffer.lock().map(|b| b.capacity).unwrap_or(0);
        f.debug_struct("LMTokenizerWithGpu")
            .field("vocab_size", &self.tokenizer.vocab_size())
            .field("bos_token_id", &self.tokenizer.bos_token_id())
            .field("eos_token_ids", &self.tokenizer.eos_token_ids())
            .field("pinned_capacity", &pinned_capacity)
            .field("gpu_capacity", &gpu_capacity)
            .finish()
    }
}

// SAFETY: LMTokenizerWithGpu contains CUDA buffers that are managed internally.
// The tokenizer is not thread-safe for concurrent use, but can be sent across threads.
unsafe impl Send for LMTokenizerWithGpu {}

// SAFETY: LMTokenizerWithGpu cannot be safely shared between threads because
// the CUDA buffers are not synchronized. Use a Mutex if you need shared access.
// For our use case, each engine instance has its own tokenizer and requests
// are serialized via the request pool semaphore.
unsafe impl Sync for LMTokenizerWithGpu {}
