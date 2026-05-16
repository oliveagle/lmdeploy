// Copyright (c) OpenMMLab. All rights reserved.
//
// TurboMind C API - Stable FFI interface for Rust/other languages
// Version: 0.1.0
//
// Usage:
//   C:  #include <turbomind_c_api.h>
//   Rust: use bindgen to generate FFI bindings, or use turbomind-sys crate

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef TM_BUILDING_DLL
    #define TM_API __declspec(dllexport)
  #else
    #define TM_API __declspec(dllimport)
  #endif
#else
  #define TM_API __attribute__((visibility("default")))
#endif

// ============================================================
// Version
// ============================================================

#define TM_VERSION_MAJOR 0
#define TM_VERSION_MINOR 1
#define TM_VERSION_PATCH 0

TM_API const char* TM_GetVersion(void);

// ============================================================
// Status / Error Codes
// ============================================================

typedef int TM_Status;

enum {
    TM_SUCCESS              = 0,
    TM_ERR_INVALID_ARG      = 1,
    TM_ERR_OOM              = 2,
    TM_ERR_TIMEOUT          = 3,
    TM_ERR_MODEL_NOT_FOUND  = 4,
    TM_ERR_DEVICE_ERROR     = 5,
    TM_ERR_INTERNAL         = 99,
};

// Get last error string (thread-local, not thread-safe)
TM_API const char* TM_GetLastError(void);

// ============================================================
// Opaque Handles
// ============================================================

typedef void* TM_Engine;
typedef void* TM_Request;
typedef void* TM_Tokenizer;

// ============================================================
// Data Types
// ============================================================

typedef enum {
    TM_FLOAT32 = 0,
    TM_FLOAT16 = 1,
    TM_BFLOAT16 = 2,
    TM_INT8    = 3,
    TM_INT32   = 4,
    TM_INT64   = 5,
} TM_DataType;

// ============================================================
// Tensor (C-compatible, caller-allocated memory)
// ============================================================

typedef struct {
    TM_DataType dtype;
    int         ndim;
    int64_t     shape[8];  // max 8 dimensions
    void*       data;      // caller-allocated, API does not free
    int         device_id; // -1 for host/CPU, >= 0 for CUDA device
} TM_Tensor;

// ============================================================
// Tensor Map (array of named tensors)
// ============================================================

typedef struct {
    const char** names;   // array of strings
    TM_Tensor*   tensors; // array of tensors
    int          count;   // number of tensors
} TM_TensorMap;

// ============================================================
// Engine Configuration
// ============================================================

typedef struct {
    int         tp_size;               // tensor parallel size
    int         max_batch_size;        // max batch size
    int         session_len;           // max context length
    float       cache_max_entry_count; // KV cache memory ratio
    int         cache_block_seq_len;   // cache block size
    int         max_prefill_iters;     // max prefill iterations
    int         enable_prefix_caching; // 0 = off, 1 = on
    int         enable_paged_kv_cache; // 0 = off, 1 = on
    int         quant_policy;          // 0 = off, 4/8 = KV cache quant bits
    const char* data_type;             // "fp16", "bf16", etc.
    int         device_ids[8];         // GPU device IDs
    int         num_devices;           // number of devices in device_ids
} TM_EngineConfig;

// ============================================================
// Generation Configuration
// ============================================================

typedef struct {
    int   max_new_tokens;
    int   min_new_tokens;
    float temperature;
    int   top_k;
    float top_p;
    float min_p;
    float repetition_penalty;
    uint64_t random_seed;
    int   output_logprobs;       // number of top logprobs per token (0 = none)
    int   output_logits;         // 0 = none, 1 = all, 2 = generation only
    int   output_last_hidden_state; // 0 = none, 1 = all, 2 = generation only
    const int* eos_ids;          // array of EOS token IDs
    int   eos_ids_count;
    const int* stop_ids;         // array of stop token IDs
    int   stop_ids_count;
    const int* bad_ids;          // array of bad token IDs
    int   bad_ids_count;
} TM_GenerationConfig;

// ============================================================
// Request State
// ============================================================

typedef enum {
    TM_STATE_INIT     = 0,
    TM_STATE_RUNNING   = 1,
    TM_STATE_FINISH    = 2,
    TM_STATE_CANCEL    = 3,
    TM_STATE_FAIL      = 4,
    TM_STATE_TOO_LONG  = 5,
    TM_STATE_BUSY      = 6,
} TM_RequestState;

// ============================================================
// Schedule Metrics
// ============================================================

typedef struct {
    int    active_seqs;
    int    max_seqs;
    int    cache_blocks_total;
    int    cache_blocks_free;
    int    cache_blocks_used;
    float  cache_usage;     // 0.0 - 1.0
    double avg_latency_ms;  // average request latency
    double avg_tps;         // average tokens/sec
} TM_ScheduleMetrics;

// ============================================================
// Request Metrics
// ============================================================

typedef struct {
    int     prompt_tokens;
    int     completion_tokens;
    double  prefill_time_ms;
    double  decode_time_ms;
    double  total_time_ms;
    double  ttft_ms;         // time to first token
} TM_RequestMetrics;

// ============================================================
// Callbacks
// ============================================================

// Called each step with new output tokens. Returns 0 to continue, non-zero to stop.
typedef int (*TM_StepCallback)(const int* token_ids, int count, TM_RequestState state, void* user_data);

// Called when request is complete with final metrics.
typedef void (*TM_DoneCallback)(TM_Status status, const TM_RequestMetrics* metrics, void* user_data);

// ============================================================
// Engine Lifecycle
// ============================================================

// Create a new engine instance.
// model_dir: path to the model directory (TurboMind format)
// config: engine configuration (copied, not held)
// devices: list of GPU device IDs (copied)
// Returns: engine handle, or NULL on failure (check TM_GetLastError)
TM_API TM_Engine TM_CreateEngine(const char* model_dir, const TM_EngineConfig* config);

// Destroy engine and free all resources.
TM_API void TM_DestroyEngine(TM_Engine engine);

// Initialize CUDA context and load model weights for device `index`.
// Call for each GPU in the tensor-parallel group.
TM_API TM_Status TM_InitDevice(TM_Engine engine, int index);

// Check if engine is a dummy node (no real weights loaded).
TM_API int TM_IsDummyNode(TM_Engine engine);

// ============================================================
// Engine Information
// ============================================================

TM_API int TM_GetVocabSize(TM_Engine engine);
TM_API int TM_GetHiddenDim(TM_Engine engine);
TM_API int TM_GetMaxBatchSize(TM_Engine engine);
TM_API int TM_GetSessionLen(TM_Engine engine);
TM_API int TM_GetNumDevices(TM_Engine engine);
TM_API TM_DataType TM_GetDataType(TM_Engine engine);

// ============================================================
// Schedule Metrics
// ============================================================

TM_API TM_Status TM_GetScheduleMetrics(TM_Engine engine, int device_index, TM_ScheduleMetrics* out_metrics);

// ============================================================
// Request Lifecycle
// ============================================================

// Create a new inference request handle.
// Returns: request handle, or NULL on failure.
TM_API TM_Request TM_CreateRequest(TM_Engine engine);

// Destroy request handle.
TM_API void TM_DestroyRequest(TM_Engine engine, TM_Request request);

// Cancel a running request.
TM_API TM_Status TM_CancelRequest(TM_Engine engine, TM_Request request);

// End a session (free sequence resources).
TM_API TM_Status TM_EndSession(TM_Engine engine, TM_Request request, uint64_t session_id);

// ============================================================
// Forward (Synchronous)
// ============================================================

// Send a forward request. Blocks until the request is complete.
// engine: engine handle
// request: request handle
// session_id: session identifier (must be unique per conversation)
// input_ids: array of token IDs
// input_len: number of tokens
// gen_cfg: generation configuration
// start: 1 if this is the first request in the session, 0 otherwise
// stop: 1 if this is the last request in the session, 0 otherwise
// out_token_ids: buffer to store generated token IDs (caller-allocated, size >= gen_cfg.max_new_tokens)
// out_count: pointer to store the actual number of generated tokens
// out_metrics: pointer to store request metrics (can be NULL)
//
// Returns: TM_SUCCESS on success, error code on failure.
TM_API TM_Status TM_Forward(
    TM_Engine             engine,
    TM_Request            request,
    uint64_t              session_id,
    const int*            input_ids,
    int                   input_len,
    const TM_GenerationConfig* gen_cfg,
    int                   start,
    int                   stop,
    int*                  out_token_ids,
    int*                  out_count,
    TM_RequestMetrics*    out_metrics
);

// ============================================================
// Forward (Asynchronous / Streaming)
// ============================================================

// Send a forward request with callbacks for streaming output.
// step_cb: called each step with new token(s). Return non-zero to stop early.
// done_cb: called when request completes (success or failure).
// user_data: passed to both callbacks.
//
// Returns: TM_SUCCESS if request was submitted, error code on failure.
// Note: The engine must be running an event loop (e.g., TM_RunEngineLoop)
// to process async requests.
TM_API TM_Status TM_ForwardAsync(
    TM_Engine             engine,
    TM_Request            request,
    uint64_t              session_id,
    const int*            input_ids,
    int                   input_len,
    const TM_GenerationConfig* gen_cfg,
    int                   start,
    int                   stop,
    TM_StepCallback       step_cb,
    TM_DoneCallback       done_cb,
    void*                 user_data
);

// ============================================================
// Engine Loop (for async processing)
// ============================================================

// Run the engine loop for one iteration (processes pending requests).
// Call this repeatedly from a background thread when using async requests.
// Returns: TM_SUCCESS if work was done, TM_ERR_INTERNAL on error.
TM_API TM_Status TM_RunEngineLoop(TM_Engine engine);

// ============================================================
// Tokenizer
// ============================================================

// Create a tokenizer handle.
TM_API TM_Tokenizer TM_CreateTokenizer(const char* model_dir);

// Destroy tokenizer handle.
TM_API void TM_DestroyTokenizer(TM_Tokenizer tokenizer);

// Encode text to token IDs.
// tokenizer: tokenizer handle
// text: input text (null-terminated)
// out_token_ids: output buffer for token IDs (caller-allocated)
// max_tokens: size of out_token_ids buffer
// out_count: actual number of tokens written
TM_API TM_Status TM_Encode(
    TM_Tokenizer tokenizer,
    const char*  text,
    int*         out_token_ids,
    int          max_tokens,
    int*         out_count
);

// Decode token IDs to text.
// tokenizer: tokenizer handle
// token_ids: input token IDs
// count: number of tokens
// out_text: output string (allocated by API, caller must TM_FreeString)
TM_API TM_Status TM_Decode(
    TM_Tokenizer tokenizer,
    const int*   token_ids,
    int          count,
    char**       out_text
);

// Free a string allocated by the API (e.g., from TM_Decode).
TM_API void TM_FreeString(char* str);

// ============================================================
// Tensor Utilities
// ============================================================

// Create a tensor from DLPack capsule (for zero-copy GPU data sharing).
// Returns: 0 on success, -1 on failure.
TM_API int TM_TensorFromDLPack(void* dlpack_capsule, TM_Tensor* out_tensor);

// Convert a tensor to DLPack capsule.
// Returns: DLPack capsule pointer (caller must release via PyCapsule destructor).
TM_API void* TM_TensorToDLPack(const TM_Tensor* tensor);

#ifdef __cplusplus
}
#endif
