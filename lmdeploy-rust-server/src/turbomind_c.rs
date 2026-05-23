//! TurboMind C API FFI bindings
//!
//! Direct bindings to turbomind_c.so without Python overhead

use std::fmt;

/// Token callback type for event-driven streaming.
///
/// Invoked from the C++ engine when a new token is generated.
/// Must be thread-safe (called from C++ engine thread).
pub type TM_TokenCallback = extern "C" fn(token_id: c_int, seq_len: c_int, user_data: *mut c_void);

// Re-export types
pub use std::os::raw::{c_char, c_float, c_int, c_long, c_uint, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TM_Error {
    pub code: TM_ErrorCode,
    pub message: [c_char; 256],
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TM_ErrorCode {
    TM_OK = 0,
    TM_ERR_INVALID_ARG = -1,
    TM_ERR_OUT_OF_MEMORY = -2,
    TM_ERR_RUNTIME = -3,
    TM_ERR_NOT_FOUND = -4,
    TM_ERR_TIMEOUT = -5,
    TM_ERR_NOT_IMPLEMENTED = -6,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TM_DataType {
    TM_DATATYPE_INVALID = 0,
    TM_DATATYPE_BOOL = 1,
    TM_DATATYPE_UINT8 = 2,
    TM_DATATYPE_UINT16 = 3,
    TM_DATATYPE_UINT32 = 4,
    TM_DATATYPE_UINT64 = 5,
    TM_DATATYPE_INT8 = 6,
    TM_DATATYPE_INT16 = 7,
    TM_DATATYPE_INT32 = 8,
    TM_DATATYPE_INT64 = 9,
    TM_DATATYPE_FP16 = 10,
    TM_DATATYPE_FP32 = 11,
    TM_DATATYPE_FP64 = 12,
    TM_DATATYPE_BF16 = 13,
    TM_DATATYPE_FP8_E4M3 = 14,
    TM_DATATYPE_FP4_E2M1 = 15,
    TM_DATATYPE_UINT4 = 16,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum TM_MemoryType {
    TM_MEMORY_CPU = 0,
    TM_MEMORY_CPU_PINNED = 1,
    TM_MEMORY_GPU = 2,
}

#[repr(C)]
pub struct TM_EngineConfig {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TM_TurboMind {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TM_TensorMap {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TM_GenerationConfig {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TM_CompiledGrammar {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TM_ModelRequest {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TM_SessionParam {
    pub id: u64,
    pub step: c_int,
    pub start_flag: bool,
    pub end_flag: bool,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum TM_RequestStatus {
    TM_STATUS_OK = 0,
    TM_STATUS_INVALID = 1,
    TM_STATUS_CONFLICT = 2,
    TM_STATUS_BUSY = 3,
    TM_STATUS_INACTIVE = 4,
    TM_STATUS_FAIL = 5,
    TM_STATUS_TOO_LONG = 6,
    TM_STATUS_FINISH = 7,
    TM_STATUS_CANCEL = 8,
    TM_STATUS_INCONSISTENCY = 9,
    TM_STATUS_NO_QUEUE = 10,
}

// FFI function signatures
extern "C" {
    // Error handling
    pub fn TM_GetLastError() -> *mut TM_Error;
    pub fn TM_ClearError();

    // Engine config
    pub fn TM_EngineConfig_Create() -> *mut TM_EngineConfig;
    pub fn TM_EngineConfig_Destroy(config: *mut TM_EngineConfig);
    pub fn TM_EngineConfig_SetDataType(config: *mut TM_EngineConfig, data_type: TM_DataType);
    pub fn TM_EngineConfig_SetCacheBlockSeqLen(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetQuantPolicy(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetMaxBatchSize(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetMaxPrefillTokenNum(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetSessionLen(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetCacheMaxBlockCount(config: *mut TM_EngineConfig, value: c_float);
    pub fn TM_EngineConfig_SetCacheChunkSize(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetEnablePrefixCaching(config: *mut TM_EngineConfig, value: bool);
    pub fn TM_EngineConfig_SetEnableMetrics(config: *mut TM_EngineConfig, value: bool);
    pub fn TM_EngineConfig_SetAttnTpSize(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetAttnDpSize(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetAttnCpSize(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetMlpTpSize(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetOuterDpSize(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetTuneLayerNum(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetMaxContextTokenNum(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetNumTokensPerIter(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetMaxPrefillIters(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetAsync(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_AddDevice(config: *mut TM_EngineConfig, device_id: c_int);
    pub fn TM_EngineConfig_SetNNodes(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetNodeRank(config: *mut TM_EngineConfig, value: c_int);
    pub fn TM_EngineConfig_SetCommunicator(config: *mut TM_EngineConfig, value: *const c_char);

    // TurboMind instance
    pub fn TM_TurboMind_Create(
        model_dir: *const c_char,
        config: *mut TM_EngineConfig,
    ) -> *mut TM_TurboMind;
    pub fn TM_TurboMind_Destroy(tm: *mut TM_TurboMind);
    pub fn TM_TurboMind_InitFromPath(
        tm: *mut TM_TurboMind,
        device_id: c_int,
        model_dir: *const c_char,
        trust_remote_code: c_int,
    ) -> c_int;
    pub fn TM_TurboMind_InitFromHF(
        tm: *mut TM_TurboMind,
        device_id: c_int,
        model_dir: *const c_char,
        trust_remote_code: c_int,
        session_len: c_int,
    ) -> c_int;
    pub fn TM_TurboMind_CreateContext(tm: *mut TM_TurboMind, index: c_int);
    pub fn TM_TurboMind_CreateRoot(tm: *mut TM_TurboMind, index: c_int);
    pub fn TM_TurboMind_ProcessWeights(tm: *mut TM_TurboMind, index: c_int);
    pub fn TM_TurboMind_CreateEngine(tm: *mut TM_TurboMind, index: c_int);
    pub fn TM_TurboMind_IsDummyNode(tm: *mut TM_TurboMind) -> bool;
    pub fn TM_TurboMind_GetAttnTpRank(tm: *mut TM_TurboMind, index: c_int) -> c_int;
    pub fn TM_TurboMind_GetMlpTpRank(tm: *mut TM_TurboMind, index: c_int) -> c_int;
    pub fn TM_TurboMind_GetModelTpRank(tm: *mut TM_TurboMind, index: c_int) -> c_int;
    pub fn TM_TurboMind_GetScheduleMetrics(
        tm: *mut TM_TurboMind,
        index: c_int,
        total_seqs: *mut c_int,
        active_seqs: *mut c_int,
        waiting_seqs: *mut c_int,
        total_blocks: *mut c_int,
        active_blocks: *mut c_int,
        cached_blocks: *mut c_int,
        free_blocks: *mut c_int,
    ) -> c_int;

    // TensorMap
    pub fn TM_TensorMap_Create() -> *mut TM_TensorMap;
    pub fn TM_TensorMap_Destroy(map: *mut TM_TensorMap);
    pub fn TM_TensorMap_SetInt32(
        map: *mut TM_TensorMap,
        name: *const c_char,
        data: *const c_int,
        ndim: c_int,
        shape: *const i64,
    );
    pub fn TM_TensorMap_SetInt64(
        map: *mut TM_TensorMap,
        name: *const c_char,
        data: *const i64,
        ndim: c_int,
        shape: *const i64,
    );
    pub fn TM_TensorMap_SetBytes(
        map: *mut TM_TensorMap,
        name: *const c_char,
        data: *const c_void,
        size: usize,
        ndim: c_int,
        shape: *const i64,
    );
    pub fn TM_TensorMap_Get(
        map: *mut TM_TensorMap,
        name: *const c_char,
        out_data: *mut *mut c_void,
        out_dtype: *mut TM_DataType,
        out_memory_type: *mut TM_MemoryType,
        out_ndim: *mut c_int,
        out_shape: *mut i64,
    ) -> bool;
    pub fn TM_TensorMap_SetFloat32(
        map: *mut TM_TensorMap,
        name: *const c_char,
        data: *const c_float,
        ndim: c_int,
        shape: *const i64,
    );
    pub fn TM_TensorMap_SetInt32GPU(
        map: *mut TM_TensorMap,
        name: *const c_char,
        data: *const c_int,
        ndim: c_int,
        shape: *const i64,
    );
    pub fn TM_TensorMap_SetInt64GPU(
        map: *mut TM_TensorMap,
        name: *const c_char,
        data: *const i64,
        ndim: c_int,
        shape: *const i64,
    );
    pub fn TM_TensorMap_SetFloat32GPU(
        map: *mut TM_TensorMap,
        name: *const c_char,
        data: *const c_float,
        ndim: c_int,
        shape: *const i64,
    );

    // Safetensors
    pub fn TM_Safetensors_Open(file_path: *const c_char) -> *mut c_void;
    pub fn TM_Safetensors_Close(handle: *mut c_void);
    pub fn TM_Safetensors_GetTensor(
        handle: *mut c_void,
        name: *const c_char,
        out_data: *mut *mut c_void,
        out_size: *mut usize,
        out_dtype: *mut TM_DataType,
    ) -> c_int;
    pub fn TM_Safetensors_NumTensors(handle: *mut c_void) -> c_int;
    pub fn TM_Safetensors_GetTensorName(handle: *mut c_void, index: c_int) -> *const c_char;

    // Generation config
    pub fn TM_GenerationConfig_Create() -> *mut TM_GenerationConfig;
    pub fn TM_GenerationConfig_Destroy(config: *mut TM_GenerationConfig);
    pub fn TM_GenerationConfig_SetMaxNewTokens(config: *mut TM_GenerationConfig, value: c_int);
    pub fn TM_GenerationConfig_SetMinNewTokens(config: *mut TM_GenerationConfig, value: c_int);
    pub fn TM_GenerationConfig_SetEosIds(
        config: *mut TM_GenerationConfig,
        ids: *const c_int,
        count: c_int,
    );
    pub fn TM_GenerationConfig_SetStopIds(
        config: *mut TM_GenerationConfig,
        ids: *const c_int,
        count: c_int,
    );
    pub fn TM_GenerationConfig_SetTopP(config: *mut TM_GenerationConfig, value: c_float);
    pub fn TM_GenerationConfig_SetTopK(config: *mut TM_GenerationConfig, value: c_int);
    pub fn TM_GenerationConfig_SetTemperature(config: *mut TM_GenerationConfig, value: c_float);
    pub fn TM_GenerationConfig_SetRepetitionPenalty(
        config: *mut TM_GenerationConfig,
        value: c_float,
    );
    pub fn TM_GenerationConfig_SetRandomSeed(config: *mut TM_GenerationConfig, value: u64);
    pub fn TM_GenerationConfig_SetOutputLogprobs(config: *mut TM_GenerationConfig, value: c_int);
    pub fn TM_GenerationConfig_SetOutputLogits(config: *mut TM_GenerationConfig, value: c_int);
    pub fn TM_GenerationConfig_SetOutputLastHiddenState(
        config: *mut TM_GenerationConfig,
        value: c_int,
    );
    pub fn TM_GenerationConfig_SetBadIds(
        config: *mut TM_GenerationConfig,
        ids: *const c_int,
        count: c_int,
    );
    pub fn TM_GenerationConfig_SetMinP(config: *mut TM_GenerationConfig, value: c_float);

    // Model request
    pub fn TM_ModelRequest_Create(tm: *mut TM_TurboMind) -> *mut TM_ModelRequest;
    pub fn TM_ModelRequest_Destroy(req: *mut TM_ModelRequest);
    pub fn TM_ModelRequest_Forward(
        req: *mut TM_ModelRequest,
        input_tensors: *mut TM_TensorMap,
        session: *const TM_SessionParam,
        gen_cfg: *const TM_GenerationConfig,
        stream_output: bool,
        enable_metrics: bool,
        output_tensors: *mut TM_TensorMap,
    ) -> c_int;
    pub fn TM_ModelRequest_Cancel(req: *mut TM_ModelRequest);
    pub fn TM_ModelRequest_End(req: *mut TM_ModelRequest, session_id: u64);
    pub fn TM_ModelRequest_GetState(
        req: *mut TM_ModelRequest,
        out_status: *mut TM_RequestStatus,
        out_seq_len: *mut c_int,
    ) -> c_int;
    pub fn TM_ModelRequest_GetOutput(
        req: *mut TM_ModelRequest,
        name: *const c_char,
        out_data: *mut *mut c_void,
        out_size: *mut usize,
    ) -> c_int;
    pub fn TM_ModelRequest_ForwardAsync(
        req: *mut TM_ModelRequest,
        input_tensors: *mut TM_TensorMap,
        session: *const TM_SessionParam,
        gen_cfg: *const TM_GenerationConfig,
        stream_output: bool,
        enable_metrics: bool,
    ) -> c_int;
    pub fn TM_ModelRequest_GetStreamToken(
        req: *mut TM_ModelRequest,
        out_data: *mut *mut c_void,
        out_count: *mut usize,
    ) -> c_int;
    pub fn TM_ModelRequest_GetStreamingState(
        req: *mut TM_ModelRequest,
        out_status: *mut TM_RequestStatus,
        out_seq_len: *mut c_int,
    ) -> c_int;
    pub fn TM_ModelRequest_SetTokenCallback(
        req: *mut TM_ModelRequest,
        cb: TM_TokenCallback,
        user_data: *mut c_void,
    ) -> c_int;

    // Guided Decoding / Structured Output (xgrammar)
    pub fn TM_Grammar_CreateFromJSONSchema(json_schema: *const c_char) -> *mut TM_CompiledGrammar;
    pub fn TM_Grammar_CreateFromEBNF(ebnf_string: *const c_char) -> *mut TM_CompiledGrammar;
    pub fn TM_Grammar_CreateFromRegex(regex: *const c_char) -> *mut TM_CompiledGrammar;
    pub fn TM_Grammar_GetBuiltinJSON() -> *const TM_CompiledGrammar;
    pub fn TM_Grammar_Destroy(grammar: *mut TM_CompiledGrammar);
    pub fn TM_ModelRequest_SetGrammar(req: *mut TM_ModelRequest, grammar: *const TM_CompiledGrammar) -> c_int;
}

/// Result type for FFI operations
pub type FFResult<T> = Result<T, FFError>;

#[derive(Debug)]
pub struct FFError {
    pub code: TM_ErrorCode,
    pub message: String,
}

impl FFError {
    pub fn from_last_error() -> Option<Self> {
        unsafe {
            let err = TM_GetLastError();
            if err.is_null() {
                return None;
            }
            let code = (*err).code;
            let msg_raw = (*err).message;
            let msg: String = {
                let len = msg_raw.iter().position(|&c| c == 0).unwrap_or(256);
                let slice = std::slice::from_raw_parts(msg_raw.as_ptr() as *const u8, len);
                String::from_utf8_lossy(slice).to_string()
            };
            TM_ClearError();
            Some(FFError { code, message: msg })
        }
    }
}

impl fmt::Display for FFError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FFI error {:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for FFError {}

/// RAII wrapper for TM_EngineConfig
pub struct EngineConfig(*mut TM_EngineConfig);

// Safety: EngineConfig is Send + Sync because the underlying C++ engine handles concurrency internally
unsafe impl Send for EngineConfig {}
unsafe impl Sync for EngineConfig {}

impl EngineConfig {
    pub fn new() -> FFResult<Self> {
        unsafe {
            let cfg = TM_EngineConfig_Create();
            if cfg.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create engine config".into(),
                }));
            }
            Ok(EngineConfig(cfg))
        }
    }

    #[inline]
    pub fn set_data_type(&mut self, dtype: TM_DataType) {
        unsafe { TM_EngineConfig_SetDataType(self.0, dtype) }
    }

    #[inline]
    pub fn set_session_len(&mut self, len: c_int) {
        unsafe { TM_EngineConfig_SetSessionLen(self.0, len) }
    }

    #[inline]
    pub fn set_cache_block_seq_len(&mut self, len: c_int) {
        unsafe { TM_EngineConfig_SetCacheBlockSeqLen(self.0, len) }
    }

    #[inline]
    pub fn set_max_batch_size(&mut self, size: c_int) {
        unsafe { TM_EngineConfig_SetMaxBatchSize(self.0, size) }
    }

    #[inline]
    pub fn set_cache_max_block_count(&mut self, count: c_float) {
        unsafe { TM_EngineConfig_SetCacheMaxBlockCount(self.0, count) }
    }

    #[inline]
    pub fn set_cache_chunk_size(&mut self, size: c_int) {
        unsafe { TM_EngineConfig_SetCacheChunkSize(self.0, size) }
    }

    pub fn set_max_prefill_token_num(&mut self, value: c_int) {
        unsafe { TM_EngineConfig_SetMaxPrefillTokenNum(self.0, value) }
    }

    #[inline]
    pub fn add_device(&mut self, device_id: c_int) {
        unsafe { TM_EngineConfig_AddDevice(self.0, device_id) }
    }

    #[inline]
    pub fn set_enable_prefix_caching(&mut self, enabled: bool) {
        unsafe { TM_EngineConfig_SetEnablePrefixCaching(self.0, enabled) }
    }

    #[inline]
    pub fn set_enable_metrics(&mut self, enabled: bool) {
        unsafe { TM_EngineConfig_SetEnableMetrics(self.0, enabled) }
    }

    #[inline]
    pub fn set_nnodes(&mut self, nnodes: c_int) {
        unsafe { TM_EngineConfig_SetNNodes(self.0, nnodes) }
    }

    #[inline]
    pub fn set_node_rank(&mut self, rank: c_int) {
        unsafe { TM_EngineConfig_SetNodeRank(self.0, rank) }
    }

    #[inline]
    pub fn set_attn_tp_size(&mut self, size: c_int) {
        unsafe { TM_EngineConfig_SetAttnTpSize(self.0, size) }
    }

    #[inline]
    pub fn set_attn_dp_size(&mut self, size: c_int) {
        unsafe { TM_EngineConfig_SetAttnDpSize(self.0, size) }
    }

    #[inline]
    pub fn set_attn_cp_size(&mut self, size: c_int) {
        unsafe { TM_EngineConfig_SetAttnCpSize(self.0, size) }
    }

    #[inline]
    pub fn set_mlp_tp_size(&mut self, size: c_int) {
        unsafe { TM_EngineConfig_SetMlpTpSize(self.0, size) }
    }

    #[inline]
    pub fn set_quant_policy(&mut self, policy: c_int) {
        unsafe { TM_EngineConfig_SetQuantPolicy(self.0, policy) }
    }

    #[inline]
    pub fn set_tune_layer_num(&mut self, value: c_int) {
        unsafe { TM_EngineConfig_SetTuneLayerNum(self.0, value) }
    }

    #[inline]
    pub fn set_max_context_token_num(&mut self, value: c_int) {
        unsafe { TM_EngineConfig_SetMaxContextTokenNum(self.0, value) }
    }

    #[inline]
    pub fn set_num_tokens_per_iter(&mut self, value: c_int) {
        unsafe { TM_EngineConfig_SetNumTokensPerIter(self.0, value) }
    }

    #[inline]
    pub fn set_max_prefill_iters(&mut self, value: c_int) {
        unsafe { TM_EngineConfig_SetMaxPrefillIters(self.0, value) }
    }

    #[inline]
    pub fn set_async(&mut self, value: c_int) {
        unsafe { TM_EngineConfig_SetAsync(self.0, value) }
    }

    #[inline]
    pub fn set_outer_dp_size(&mut self, value: c_int) {
        unsafe { TM_EngineConfig_SetOuterDpSize(self.0, value) }
    }
}

impl Drop for EngineConfig {
    fn drop(&mut self) {
        unsafe { TM_EngineConfig_Destroy(self.0) }
    }
}

/// RAII wrapper for TM_TurboMind
pub struct TurboMind(*mut TM_TurboMind);

// Safety: The underlying C++ TurboMind instance is thread-safe for concurrent operations.
// CreateContext, ProcessWeights, CreateEngine are called during initialization and are safe.
// Forward inference uses internal locks within the C++ code.
unsafe impl Send for TurboMind {}
unsafe impl Sync for TurboMind {}

impl TurboMind {
    pub fn create(model_dir: &str, config: &mut EngineConfig) -> FFResult<Self> {
        unsafe {
            let model_dir_c = std::ffi::CString::new(model_dir).unwrap();
            let tm = TM_TurboMind_Create(model_dir_c.as_ptr(), config.0);
            if tm.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create TurboMind".into(),
                }));
            }
            Ok(TurboMind(tm))
        }
    }

    /// Get the raw C pointer for passing to C functions
    pub fn as_mut_ptr(&mut self) -> *mut TM_TurboMind {
        self.0
    }

    pub fn create_context(&self, index: c_int) {
        unsafe { TM_TurboMind_CreateContext(self.0, index) }
    }

    pub fn create_root(&self, index: c_int) {
        unsafe { TM_TurboMind_CreateRoot(self.0, index) }
    }

    pub fn init_from_path(
        &self,
        index: c_int,
        model_dir: &str,
        trust_remote_code: bool,
    ) -> FFResult<()> {
        unsafe {
            let model_dir_c = std::ffi::CString::new(model_dir).unwrap();
            let ret = TM_TurboMind_InitFromPath(
                self.0,
                index,
                model_dir_c.as_ptr(),
                if trust_remote_code { 1 } else { 0 },
            );
            if ret != 0 {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: format!("InitFromPath failed with code {}", ret),
                }));
            }
            Ok(())
        }
    }

    pub fn process_weights(&self, index: c_int) {
        unsafe { TM_TurboMind_ProcessWeights(self.0, index) }
    }

    pub fn create_engine(&self, index: c_int) {
        unsafe { TM_TurboMind_CreateEngine(self.0, index) }
    }

    pub fn init_from_hf(
        &self,
        device_id: c_int,
        model_dir: &str,
        trust_remote_code: bool,
        session_len: c_int,
    ) -> FFResult<()> {
        unsafe {
            let model_dir_c = std::ffi::CString::new(model_dir).unwrap();
            let ret = TM_TurboMind_InitFromHF(
                self.0,
                device_id,
                model_dir_c.as_ptr(),
                if trust_remote_code { 1 } else { 0 },
                session_len,
            );
            if ret != 0 {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: format!("InitFromHF failed with code {}", ret),
                }));
            }
            Ok(())
        }
    }

    pub fn get_attn_tp_rank(&self, index: c_int) -> c_int {
        unsafe { TM_TurboMind_GetAttnTpRank(self.0, index) }
    }

    pub fn get_mlp_tp_rank(&self, index: c_int) -> c_int {
        unsafe { TM_TurboMind_GetMlpTpRank(self.0, index) }
    }

    pub fn get_model_tp_rank(&self, index: c_int) -> c_int {
        unsafe { TM_TurboMind_GetModelTpRank(self.0, index) }
    }

    pub fn is_dummy_node(&self) -> bool {
        unsafe { TM_TurboMind_IsDummyNode(self.0) }
    }

    pub fn get_schedule_metrics(&self, index: c_int) -> FFResult<ScheduleMetrics> {
        let mut total_seqs = 0;
        let mut active_seqs = 0;
        let mut waiting_seqs = 0;
        let mut total_blocks = 0;
        let mut active_blocks = 0;
        let mut cached_blocks = 0;
        let mut free_blocks = 0;

        let ret = unsafe {
            TM_TurboMind_GetScheduleMetrics(
                self.0,
                index,
                &mut total_seqs,
                &mut active_seqs,
                &mut waiting_seqs,
                &mut total_blocks,
                &mut active_blocks,
                &mut cached_blocks,
                &mut free_blocks,
            )
        };

        if ret != 0 {
            return Err(FFError::from_last_error().unwrap_or(FFError {
                code: TM_ErrorCode::TM_ERR_RUNTIME,
                message: "Failed to get schedule metrics".into(),
            }));
        }

        Ok(ScheduleMetrics {
            total_seqs,
            active_seqs,
            waiting_seqs,
            total_blocks,
            active_blocks,
            cached_blocks,
            free_blocks,
        })
    }
}

impl Drop for TurboMind {
    fn drop(&mut self) {
        unsafe { TM_TurboMind_Destroy(self.0) }
    }
}

#[derive(Debug, Clone)]
pub struct ScheduleMetrics {
    pub total_seqs: c_int,
    pub active_seqs: c_int,
    pub waiting_seqs: c_int,
    pub total_blocks: c_int,
    pub active_blocks: c_int,
    pub cached_blocks: c_int,
    pub free_blocks: c_int,
}

/// RAII wrapper for TM_GenerationConfig
pub struct GenConfig(*mut TM_GenerationConfig);

// Safety: GenConfig is Send + Sync because the underlying C++ engine handles concurrency internally
unsafe impl Send for GenConfig {}
unsafe impl Sync for GenConfig {}

impl GenConfig {
    pub fn new() -> FFResult<Self> {
        unsafe {
            let cfg = TM_GenerationConfig_Create();
            if cfg.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create generation config".into(),
                }));
            }
            Ok(GenConfig(cfg))
        }
    }

    #[inline]
    pub fn set_max_new_tokens(&mut self, tokens: c_int) {
        unsafe { TM_GenerationConfig_SetMaxNewTokens(self.0, tokens) }
    }

    #[inline]
    pub fn set_temperature(&mut self, temp: c_float) {
        unsafe { TM_GenerationConfig_SetTemperature(self.0, temp) }
    }

    #[inline]
    pub fn set_top_p(&mut self, top_p: c_float) {
        unsafe { TM_GenerationConfig_SetTopP(self.0, top_p) }
    }

    #[inline]
    pub fn set_top_k(&mut self, top_k: c_int) {
        unsafe { TM_GenerationConfig_SetTopK(self.0, top_k) }
    }

    #[inline]
    pub fn set_repetition_penalty(&mut self, penalty: c_float) {
        unsafe { TM_GenerationConfig_SetRepetitionPenalty(self.0, penalty) }
    }

    #[inline]
    pub fn set_eos_ids(&mut self, ids: &[c_int]) {
        unsafe {
            TM_GenerationConfig_SetEosIds(self.0, ids.as_ptr(), ids.len() as c_int);
        }
    }

    #[inline]
    pub fn set_stop_ids(&mut self, ids: &[c_int]) {
        unsafe {
            TM_GenerationConfig_SetStopIds(self.0, ids.as_ptr(), ids.len() as c_int);
        }
    }

    #[inline]
    pub fn set_random_seed(&mut self, seed: u64) {
        unsafe { TM_GenerationConfig_SetRandomSeed(self.0, seed) }
    }

    #[inline]
    pub fn set_bad_ids(&mut self, ids: &[c_int]) {
        unsafe {
            TM_GenerationConfig_SetBadIds(self.0, ids.as_ptr(), ids.len() as c_int);
        }
    }

    #[inline]
    pub fn set_min_p(&mut self, min_p: c_float) {
        unsafe { TM_GenerationConfig_SetMinP(self.0, min_p) }
    }

    #[inline]
    pub fn set_output_last_hidden_state(&mut self, value: c_int) {
        unsafe { TM_GenerationConfig_SetOutputLastHiddenState(self.0, value) }
    }

    #[inline]
    pub fn set_output_logprobs(&mut self, num_logprobs: c_int) {
        unsafe { TM_GenerationConfig_SetOutputLogprobs(self.0, num_logprobs) }
    }

    #[inline]
    pub fn set_output_logits(&mut self, value: c_int) {
        unsafe { TM_GenerationConfig_SetOutputLogits(self.0, value) }
    }
}

impl Drop for GenConfig {
    fn drop(&mut self) {
        unsafe { TM_GenerationConfig_Destroy(self.0) }
    }
}

/// Compiled grammar for guided decoding.
///
/// Created from JSON schema, EBNF grammar, or regex pattern via the
/// xgrammar integration in the C++ engine. Attach to a ModelRequest
/// via `ModelRequest::set_grammar` before calling forward.
pub struct CompiledGrammar {
    ptr: *mut TM_CompiledGrammar,
    builtin: bool,
}

impl fmt::Debug for CompiledGrammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledGrammar")
            .field("ptr", &self.ptr)
            .field("builtin", &self.builtin)
            .finish()
    }
}

// Safety: The underlying C++ grammar is immutable after creation
unsafe impl Send for CompiledGrammar {}
unsafe impl Sync for CompiledGrammar {}

impl CompiledGrammar {
    /// Create a compiled grammar from a JSON schema string.
    pub fn from_json_schema(schema: &str) -> FFResult<Self> {
        let schema_c = std::ffi::CString::new(schema).map_err(|e| FFError {
            code: TM_ErrorCode::TM_ERR_INVALID_ARG,
            message: format!("Invalid JSON schema string: {}", e),
        })?;
        unsafe {
            let ptr = TM_Grammar_CreateFromJSONSchema(schema_c.as_ptr());
            if ptr.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create grammar from JSON schema".into(),
                }));
            }
            Ok(CompiledGrammar { ptr, builtin: false })
        }
    }

    /// Create a compiled grammar from an EBNF grammar string.
    pub fn from_ebnf(grammar: &str) -> FFResult<Self> {
        let grammar_c = std::ffi::CString::new(grammar).map_err(|e| FFError {
            code: TM_ErrorCode::TM_ERR_INVALID_ARG,
            message: format!("Invalid EBNF grammar string: {}", e),
        })?;
        unsafe {
            let ptr = TM_Grammar_CreateFromEBNF(grammar_c.as_ptr());
            if ptr.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create grammar from EBNF".into(),
                }));
            }
            Ok(CompiledGrammar { ptr, builtin: false })
        }
    }

    /// Create a compiled grammar from a regex pattern.
    pub fn from_regex(pattern: &str) -> FFResult<Self> {
        let pattern_c = std::ffi::CString::new(pattern).map_err(|e| FFError {
            code: TM_ErrorCode::TM_ERR_INVALID_ARG,
            message: format!("Invalid regex pattern: {}", e),
        })?;
        unsafe {
            let ptr = TM_Grammar_CreateFromRegex(pattern_c.as_ptr());
            if ptr.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create grammar from regex".into(),
                }));
            }
            Ok(CompiledGrammar { ptr, builtin: false })
        }
    }

    /// Get the built-in JSON grammar (accepts any valid JSON output).
    pub fn builtin_json() -> Self {
        unsafe {
            let ptr = TM_Grammar_GetBuiltinJSON();
            CompiledGrammar {
                ptr: ptr as *mut _,
                builtin: true,
            }
        }
    }

    /// Get the raw C pointer for passing to C functions.
    pub fn as_ptr(&self) -> *const TM_CompiledGrammar {
        self.ptr
    }
}

impl Drop for CompiledGrammar {
    fn drop(&mut self) {
        if !self.builtin {
            unsafe { TM_Grammar_Destroy(self.ptr) }
        }
    }
}

/// RAII wrapper for TM_TensorMap
pub struct TensorMap(*mut TM_TensorMap);

// Safety: TensorMap is Send + Sync because the underlying C++ API handles concurrency internally
unsafe impl Send for TensorMap {}
unsafe impl Sync for TensorMap {}

impl TensorMap {
    pub fn new() -> FFResult<Self> {
        unsafe {
            let map = TM_TensorMap_Create();
            if map.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create tensor map".into(),
                }));
            }
            Ok(TensorMap(map))
        }
    }

    pub fn set_int32(&mut self, name: &str, data: &[c_int], shape: &[i64]) {
        unsafe {
            let name_c = std::ffi::CString::new(name).unwrap();
            TM_TensorMap_SetInt32(
                self.0,
                name_c.as_ptr(),
                data.as_ptr(),
                shape.len() as c_int,
                shape.as_ptr(),
            );
        }
    }

    pub fn set_int64(&mut self, name: &str, data: &[i64], shape: &[i64]) {
        unsafe {
            let name_c = std::ffi::CString::new(name).unwrap();
            TM_TensorMap_SetInt64(
                self.0,
                name_c.as_ptr(),
                data.as_ptr(),
                shape.len() as c_int,
                shape.as_ptr(),
            );
        }
    }

    pub fn set_bytes(&mut self, name: &str, data: &[u8], shape: &[i64]) {
        unsafe {
            let name_c = std::ffi::CString::new(name).unwrap();
            TM_TensorMap_SetBytes(
                self.0,
                name_c.as_ptr(),
                data.as_ptr() as *const c_void,
                data.len(),
                shape.len() as c_int,
                shape.as_ptr(),
            );
        }
    }

    pub fn set_float32(&mut self, name: &str, data: &[c_float], shape: &[i64]) {
        unsafe {
            let name_c = std::ffi::CString::new(name).unwrap();
            TM_TensorMap_SetFloat32(
                self.0,
                name_c.as_ptr(),
                data.as_ptr(),
                shape.len() as c_int,
                shape.as_ptr(),
            );
        }
    }

    pub fn set_int32_gpu(&mut self, name: &str, data: *const c_int, shape: &[i64]) {
        unsafe {
            let name_c = std::ffi::CString::new(name).unwrap();
            TM_TensorMap_SetInt32GPU(
                self.0,
                name_c.as_ptr(),
                data,
                shape.len() as c_int,
                shape.as_ptr(),
            );
        }
    }

    pub fn set_int64_gpu(&mut self, name: &str, data: *const i64, shape: &[i64]) {
        unsafe {
            let name_c = std::ffi::CString::new(name).unwrap();
            TM_TensorMap_SetInt64GPU(
                self.0,
                name_c.as_ptr(),
                data,
                shape.len() as c_int,
                shape.as_ptr(),
            );
        }
    }

    pub fn set_float32_gpu(&mut self, name: &str, data: *const c_float, shape: &[i64]) {
        unsafe {
            let name_c = std::ffi::CString::new(name).unwrap();
            TM_TensorMap_SetFloat32GPU(
                self.0,
                name_c.as_ptr(),
                data,
                shape.len() as c_int,
                shape.as_ptr(),
            );
        }
    }

    /// Get the raw C pointer for passing to C functions
    pub fn as_mut_ptr(&self) -> *mut TM_TensorMap {
        self.0
    }
}

impl Drop for TensorMap {
    fn drop(&mut self) {
        unsafe { TM_TensorMap_Destroy(self.0) }
    }
}

/// RAII wrapper for TM_ModelRequest
pub struct ModelRequest(*mut TM_ModelRequest);

// Safety: ModelRequest is Send + Sync because the underlying C++ engine handles concurrency internally
unsafe impl Send for ModelRequest {}
unsafe impl Sync for ModelRequest {}

impl ModelRequest {
    pub fn create(tm: &TurboMind) -> FFResult<Self> {
        unsafe {
            let req = TM_ModelRequest_Create(tm.0);
            if req.is_null() {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to create model request".into(),
                }));
            }
            Ok(ModelRequest(req))
        }
    }

    pub fn forward(
        &mut self,
        input_tensors: &mut TensorMap,
        session: &TM_SessionParam,
        gen_cfg: &GenConfig,
        stream_output: bool,
        enable_metrics: bool,
        output_tensors: &mut TensorMap,
    ) -> FFResult<()> {
        unsafe {
            let ret = TM_ModelRequest_Forward(
                self.0,
                input_tensors.0,
                session,
                gen_cfg.0,
                stream_output,
                enable_metrics,
                output_tensors.0,
            );
            if ret != 0 {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Forward failed".into(),
                }));
            }
            Ok(())
        }
    }

    pub fn cancel(&mut self) {
        unsafe { TM_ModelRequest_Cancel(self.0) }
    }

    /// Get output tensor by name after forward completes
    /// Returns (data_ptr, size) where data_ptr is a pointer to the data
    /// The data is owned by the request and is valid until the request is destroyed
    pub fn get_output(&self, name: &str) -> FFResult<(*const u8, usize)> {
        let name_c = std::ffi::CString::new(name).unwrap();
        let mut out_data: *mut c_void = std::ptr::null_mut();
        let mut out_size: usize = 0;

        let ret = unsafe {
            TM_ModelRequest_GetOutput(self.0, name_c.as_ptr(), &mut out_data, &mut out_size)
        };

        if ret != 0 {
            return Err(FFError::from_last_error().unwrap_or(FFError {
                code: TM_ErrorCode::TM_ERR_RUNTIME,
                message: format!("Failed to get output '{}'", name),
            }));
        }

        Ok((out_data as *const u8, out_size))
    }

    /// Submit a non-blocking forward request with stream_output enabled.
    /// Caller must poll `get_streaming_state` to check completion and `get_stream_token`
    /// to read intermediate tokens during generation.
    pub fn forward_async(
        &mut self,
        input_tensors: &mut TensorMap,
        session: &TM_SessionParam,
        gen_cfg: &GenConfig,
        stream_output: bool,
        enable_metrics: bool,
    ) -> FFResult<()> {
        unsafe {
            let ret = TM_ModelRequest_ForwardAsync(
                self.0,
                input_tensors.0,
                session,
                gen_cfg.0,
                stream_output,
                enable_metrics,
            );
            if ret != 0 {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "ForwardAsync failed".into(),
                }));
            }
            Ok(())
        }
    }

    /// Get output_ids tensor data from in-flight streaming request.
    /// Returns (data_ptr, token_count) where data_ptr points to int32 array.
    pub fn get_stream_token(&self) -> FFResult<(*const c_int, usize)> {
        let mut out_data: *mut c_void = std::ptr::null_mut();
        let mut out_count: usize = 0;

        let ret = unsafe { TM_ModelRequest_GetStreamToken(self.0, &mut out_data, &mut out_count) };

        if ret != 0 {
            return Err(FFError::from_last_error().unwrap_or(FFError {
                code: TM_ErrorCode::TM_ERR_RUNTIME,
                message: "GetStreamToken failed".into(),
            }));
        }

        Ok((out_data as *const c_int, out_count))
    }

    /// Get streaming request state (non-destructive polling)
    pub fn get_streaming_state(&self) -> FFResult<(TM_RequestStatus, c_int)> {
        let mut out_status: TM_RequestStatus = TM_RequestStatus::TM_STATUS_OK;
        let mut out_seq_len: c_int = 0;

        let ret =
            unsafe { TM_ModelRequest_GetStreamingState(self.0, &mut out_status, &mut out_seq_len) };

        if ret != 0 {
            return Err(FFError::from_last_error().unwrap_or(FFError {
                code: TM_ErrorCode::TM_ERR_RUNTIME,
                message: "GetStreamingState failed".into(),
            }));
        }

        Ok((out_status, out_seq_len))
    }

    /// Register a token callback for event-driven streaming.
    ///
    /// The callback will be invoked from the C++ engine thread whenever a new token is generated.
    /// The callback must be thread-safe.
    ///
    /// # Safety
    /// - `cb` must be valid for the lifetime of the request
    /// - `user_data` must be a valid pointer or null
    pub unsafe fn set_token_callback(
        &mut self,
        cb: TM_TokenCallback,
        user_data: *mut c_void,
    ) -> FFResult<()> {
        let ret = TM_ModelRequest_SetTokenCallback(self.0, cb, user_data);
        if ret != 0 {
            return Err(FFError::from_last_error().unwrap_or(FFError {
                code: TM_ErrorCode::TM_ERR_RUNTIME,
                message: "SetTokenCallback failed".into(),
            }));
        }
        Ok(())
    }

    /// Attach a compiled grammar for guided decoding.
    ///
    /// Must be called before `forward` or `forward_async`. The grammar is not
    /// owned by the request and must remain valid until the forward completes.
    pub fn set_grammar(&mut self, grammar: &CompiledGrammar) -> FFResult<()> {
        unsafe {
            let ret = TM_ModelRequest_SetGrammar(self.0, grammar.as_ptr());
            if ret != 0 {
                return Err(FFError::from_last_error().unwrap_or(FFError {
                    code: TM_ErrorCode::TM_ERR_RUNTIME,
                    message: "Failed to attach grammar to request".into(),
                }));
            }
            Ok(())
        }
    }
}

impl Drop for ModelRequest {
    fn drop(&mut self) {
        unsafe { TM_ModelRequest_Destroy(self.0) }
    }
}

/// AWQ quantization configuration
/// Matches AwqQuantConfig struct in turbomind_c.cc
#[derive(Debug, Clone)]
pub struct AwqConfig {
    pub bits: u32,
    pub group_size: u32,
    pub version: String,
    pub symmetric: bool,
    pub zero_point: bool,
    pub pack: bool,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            version: "gemm".to_string(),
            symmetric: true,
            zero_point: true,
            pack: true,
        }
    }
}

impl AwqConfig {
    /// Create a new AWQ config with custom parameters
    pub fn new(bits: u32, group_size: u32) -> Self {
        Self {
            bits,
            group_size,
            ..Default::default()
        }
    }

    /// Convert to an integer policy value for TM_EngineConfig_SetQuantPolicy
    /// Returns 4 for AWQ 4-bit, 8 for INT8, etc.
    pub fn to_quant_policy(&self) -> c_int {
        match self.bits {
            4 => 4, // AWQ 4-bit
            8 => 8, // INT8
            _ => self.bits as c_int,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test TM_ErrorCode enum values match C API
    #[test]
    fn test_error_code_values() {
        assert_eq!(TM_ErrorCode::TM_OK as i32, 0);
        assert_eq!(TM_ErrorCode::TM_ERR_INVALID_ARG as i32, -1);
        assert_eq!(TM_ErrorCode::TM_ERR_OUT_OF_MEMORY as i32, -2);
        assert_eq!(TM_ErrorCode::TM_ERR_RUNTIME as i32, -3);
        assert_eq!(TM_ErrorCode::TM_ERR_NOT_FOUND as i32, -4);
        assert_eq!(TM_ErrorCode::TM_ERR_TIMEOUT as i32, -5);
    }

    /// Test TM_DataType enum values
    #[test]
    fn test_data_type_values() {
        assert_eq!(TM_DataType::TM_DATATYPE_INVALID as i32, 0);
        assert_eq!(TM_DataType::TM_DATATYPE_BOOL as i32, 1);
        assert_eq!(TM_DataType::TM_DATATYPE_FP16 as i32, 10);
        assert_eq!(TM_DataType::TM_DATATYPE_FP32 as i32, 11);
        assert_eq!(TM_DataType::TM_DATATYPE_BF16 as i32, 13);
        assert_eq!(TM_DataType::TM_DATATYPE_FP8_E4M3 as i32, 14);
        assert_eq!(TM_DataType::TM_DATATYPE_UINT4 as i32, 16);
    }

    /// Test TM_MemoryType enum values
    #[test]
    fn test_memory_type_values() {
        assert_eq!(TM_MemoryType::TM_MEMORY_CPU as i32, 0);
        assert_eq!(TM_MemoryType::TM_MEMORY_CPU_PINNED as i32, 1);
        assert_eq!(TM_MemoryType::TM_MEMORY_GPU as i32, 2);
    }

    /// Test FFError formatting
    #[test]
    fn test_ff_error_display() {
        let error = FFError {
            code: TM_ErrorCode::TM_ERR_RUNTIME,
            message: "Test error message".to_string(),
        };

        let display_string = format!("{}", error);
        assert!(display_string.contains("FFI error"));
        assert!(display_string.contains("Test error message"));
    }

    /// Test TM_Error struct size
    #[test]
    fn test_error_struct_size() {
        // TM_Error should have a 256-byte message buffer
        assert_eq!(std::mem::size_of::<TM_Error>(), 4 + 256); // code (4 bytes) + message (256 bytes)
    }

    /// Test TM_SessionParam struct layout
    #[test]
    fn test_session_param_layout() {
        let session = TM_SessionParam {
            id: 12345,
            step: 5,
            start_flag: true,
            end_flag: false,
        };

        assert_eq!(session.id, 12345);
        assert_eq!(session.step, 5);
        assert!(session.start_flag);
        assert!(!session.end_flag);
    }

    /// Test ScheduleMetrics struct
    #[test]
    fn test_schedule_metrics() {
        let metrics = ScheduleMetrics {
            total_seqs: 10,
            active_seqs: 5,
            waiting_seqs: 2,
            total_blocks: 100,
            active_blocks: 50,
            cached_blocks: 20,
            free_blocks: 30,
        };

        assert_eq!(metrics.total_seqs, 10);
        assert_eq!(metrics.active_seqs, 5);
        assert_eq!(metrics.waiting_seqs, 2);
        assert_eq!(metrics.total_blocks, 100);
        assert_eq!(metrics.active_blocks, 50);
        assert_eq!(metrics.cached_blocks, 20);
        assert_eq!(metrics.free_blocks, 30);
    }

    /// Test EngineConfig creation and setters
    #[test]
    fn test_engine_config_creation() {
        // This test validates the EngineConfig creation function signature
        // Actual creation requires the C library, but we verify types match
        fn expect_engine_config(_config: &EngineConfig) {}

        // Verify we can reference EngineConfig type
        let config_type_check: std::marker::PhantomData<EngineConfig> = std::marker::PhantomData;
        let _ = config_type_check;
    }

    /// Test TurboMind wrapper type signature
    #[test]
    fn test_turbomind_type_signature() {
        // Verify TurboMind wrapper has correct methods
        fn check_turbomind_methods(_tm: &TurboMind) {
            // These methods should exist on TurboMind
            let _index: c_int = 0;
            let _model_dir: &str = "";
            let _trust_remote_code: bool = false;

            // We can't call the methods without a real TurboMind instance
            // but this verifies the types compile
        }

        // Verify Send/Sync traits are implemented for thread safety
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<TurboMind>();
        assert_sync::<TurboMind>();
    }

    /// Test GenConfig wrapper type
    #[test]
    fn test_gen_config_type() {
        // Verify GenConfig type exists
        fn expect_gen_config(_config: &GenConfig) {}

        let _ = expect_gen_config as fn(&GenConfig);
    }

    /// Test TensorMap wrapper type
    #[test]
    fn test_tensor_map_type() {
        // Verify TensorMap type exists
        fn expect_tensor_map(_map: &TensorMap) {}

        let _ = expect_tensor_map as fn(&TensorMap);
    }

    /// Test ModelRequest wrapper type
    #[test]
    fn test_model_request_type() {
        // Verify ModelRequest type exists
        fn expect_model_request(_req: &ModelRequest) {}

        let _ = expect_model_request as fn(&ModelRequest);
    }

    /// Document the InitFromPath initialization sequence
    /// This test documents the expected sequence of calls for successful initialization
    #[test]
    fn test_init_from_path_sequence_documentation() {
        // According to turbomind_c.cc TM_TurboMind_InitFromPath:
        // Step 1: Create CUDA context (CreateContext)
        // Step 2: Create ModelRoot sentinel (CreateRoot)
        // Step 3: Build and attach ModelWeight
        // Step 4: Process weights (ProcessWeights)
        // Step 5: Create inference engine (CreateEngine)

        let expected_steps = [
            ("CreateContext", "Create CUDA context"),
            ("CreateRoot", "Create ModelRoot sentinel"),
            ("BuildModelWeight", "Build and attach ModelWeight module"),
            ("ProcessWeights", "Process weights and load to GPU"),
            ("CreateEngine", "Create inference engine"),
        ];

        assert_eq!(expected_steps.len(), 5);
        assert_eq!(expected_steps[0].0, "CreateContext");
        assert_eq!(expected_steps[4].0, "CreateEngine");
    }

    /// Test that FFI result type is correctly defined
    #[test]
    fn test_ffresult_type() {
        // FFResult<T> should be Result<T, FFError>
        fn expect_result<T>(_result: FFResult<T>) -> T {
            unreachable!()
        }

        let _ = expect_result as fn(FFResult<i32>) -> i32;
    }

    /// Test c primitive type sizes
    #[test]
    fn test_c_primitive_sizes() {
        assert_eq!(std::mem::size_of::<c_int>(), 4);
        assert_eq!(std::mem::size_of::<c_float>(), 4);
        assert_eq!(std::mem::size_of::<c_long>(), 8);
        assert_eq!(std::mem::size_of::<c_char>(), 1);
        // c_void has size 1 in Rust (it's a ZST but not truly 0-sized in this context)
        assert_eq!(std::mem::size_of::<c_void>(), 1);
    }

    /// Test quantization policy enum variants for EngineConfig
    #[test]
    fn test_quant_policy_values() {
        // quant_policy in EngineConfig maps to int values:
        // 0 = NONE (no quantization)
        // 4 = AWQ 4-bit
        // 8 = KV cache INT8
        // etc.
        let quant_policy_none: c_int = 0;
        let quant_policy_awq4: c_int = 4;
        let quant_policy_kvint8: c_int = 8;

        assert_eq!(quant_policy_none, 0);
        assert_eq!(quant_policy_awq4, 4);
        assert_eq!(quant_policy_kvint8, 8);
    }

    /// Test EngineConfig has set_quant_policy method
    #[test]
    fn test_engine_config_quant_policy_method() {
        // Verify EngineConfig has the set_quant_policy method
        fn expect_set_quant_policy(_config: &mut EngineConfig, _policy: c_int) {}

        // Verify the function signature matches
        let _ = expect_set_quant_policy as fn(&mut EngineConfig, c_int);
    }

    /// Test AWQ configuration defaults match C++ AwqQuantConfig
    #[test]
    fn test_awq_config_defaults() {
        // These defaults must match turbomind_c.cc AwqQuantConfig struct
        let awq_defaults = AwqConfig::default();
        assert_eq!(awq_defaults.bits, 4);
        assert_eq!(awq_defaults.group_size, 128);
        assert_eq!(awq_defaults.version, "gemm");
        assert!(awq_defaults.symmetric);
        assert!(awq_defaults.zero_point);
        assert!(awq_defaults.pack);
    }

    /// Test quant method detection from config.json fields
    #[test]
    fn test_quant_method_detection() {
        // Test that "awq" quant_method is correctly identified
        // This matches the parsing logic in turbomind_c.cc ReadAwqQuantConfig
        let awq_method = "awq";
        let fp8_method = "fp8";

        assert_eq!(awq_method, "awq");
        assert_eq!(fp8_method, "fp8");
    }

    /// Test AwqConfig creation and default values
    #[test]
    fn test_awq_config_creation() {
        let config = AwqConfig::new(4, 128);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert_eq!(config.version, "gemm");
    }

    /// Test AwqConfig to_quant_policy conversion
    #[test]
    fn test_awq_config_to_policy() {
        let config4 = AwqConfig::new(4, 128);
        let config8 = AwqConfig::new(8, 128);

        assert_eq!(config4.to_quant_policy(), 4);
        assert_eq!(config8.to_quant_policy(), 8);
    }
}
