// Copyright (c) OpenMMLab. All rights reserved.
// C API for TurboMind inference engine
// Allows direct FFI integration from Rust without Python

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "src/turbomind/core/data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Forward declarations
// ============================================================

typedef struct TM_TurboMind TM_TurboMind;
typedef struct TM_TensorMap TM_TensorMap;
typedef struct TM_ModelRequest TM_ModelRequest;

// ============================================================
// Error handling
// ============================================================

typedef enum {
    TM_OK = 0,
    TM_ERR_INVALID_ARG = -1,
    TM_ERR_OUT_OF_MEMORY = -2,
    TM_ERR_RUNTIME = -3,
    TM_ERR_NOT_FOUND = -4,
    TM_ERR_TIMEOUT = -5,
} TM_ErrorCode;

typedef struct {
    TM_ErrorCode code;
    char message[256];
} TM_Error;

// Get last error (thread-local). Returns NULL if no error.
TM_Error* TM_GetLastError(void);

// Clear error state
void TM_ClearError(void);

// ============================================================
// Data types
// ============================================================

typedef enum {
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
} TM_DataType;

typedef enum {
    TM_MEMORY_CPU = 0,
    TM_MEMORY_CPU_PINNED = 1,
    TM_MEMORY_GPU = 2,
} TM_MemoryType;

// ============================================================
// Engine configuration
// ============================================================

// Opaque handle for engine config
typedef struct TM_EngineConfig TM_EngineConfig;

TM_EngineConfig* TM_EngineConfig_Create(void);
void TM_EngineConfig_Destroy(TM_EngineConfig* config);

// Configure all ENGINE_FIELDS from engine_config.h
void TM_EngineConfig_SetDataType(TM_EngineConfig* config, TM_DataType data_type);
void TM_EngineConfig_SetCacheBlockSeqLen(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetQuantPolicy(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetTuneLayerNum(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetMaxBatchSize(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetMaxPrefillTokenNum(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetMaxContextTokenNum(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetSessionLen(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetCacheMaxBlockCount(TM_EngineConfig* config, float value);
void TM_EngineConfig_SetCacheChunkSize(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetEnablePrefixCaching(TM_EngineConfig* config, bool value);
void TM_EngineConfig_SetEnableMetrics(TM_EngineConfig* config, bool value);
void TM_EngineConfig_SetNumTokensPerIter(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetMaxPrefillIters(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetAsync(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetOuterDpSize(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetAttnDpSize(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetAttnTpSize(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetAttnCpSize(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetMlpTpSize(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetNNodes(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetNodeRank(TM_EngineConfig* config, int value);
void TM_EngineConfig_SetCommunicator(TM_EngineConfig* config, const char* value);

// Device management - append devices to config
void TM_EngineConfig_AddDevice(TM_EngineConfig* config, int device_id);

// ============================================================
// TurboMind instance
// ============================================================

// Create TurboMind instance from model directory and config
// Returns NULL on error (call TM_GetLastError for details)
TM_TurboMind* TM_TurboMind_Create(const char* model_dir, TM_EngineConfig* config);
void TM_TurboMind_Destroy(TM_TurboMind* tm);

// Initialize the engine (without loading weights)
// Returns 0 on success, negative on error
int TM_TurboMind_InitFromPath(TM_TurboMind* tm, int device_id, const char* model_dir, int trust_remote_code);

// ============================================================
// Safetensors file handling
// ============================================================

// Open a safetensors file for reading
void* TM_Safetensors_Open(const char* file_path);
void TM_Safetensors_Close(void* handle);
int TM_Safetensors_GetTensor(
    void* handle,
    const char* name,
    void** out_data,
    size_t* out_size,
    int* out_ndim,
    int64_t* out_shape,
    TM_DataType* out_dtype);
int TM_Safetensors_NumTensors(void* handle);

// Initialization sequence (must be called in order)
void TM_TurboMind_CreateContext(TM_TurboMind* tm, int index);
void TM_TurboMind_CreateRoot(TM_TurboMind* tm, int index);
void TM_TurboMind_ProcessWeights(TM_TurboMind* tm, int index);
void TM_TurboMind_CreateEngine(TM_TurboMind* tm, int index);

// Check if this is a dummy node (no GPU assigned)
bool TM_TurboMind_IsDummyNode(TM_TurboMind* tm);

// Get TP ranks for a device index
int TM_TurboMind_GetAttnTpRank(TM_TurboMind* tm, int index);
int TM_TurboMind_GetMlpTpRank(TM_TurboMind* tm, int index);
int TM_TurboMind_GetModelTpRank(TM_TurboMind* tm, int index);

// Get schedule metrics
// Returns 0 on success, -1 on error
int TM_TurboMind_GetScheduleMetrics(
    TM_TurboMind* tm,
    int index,
    int* total_seqs,
    int* active_seqs,
    int* waiting_seqs,
    int* total_blocks,
    int* active_blocks,
    int* cached_blocks,
    int* free_blocks);

// ============================================================
// Tensor / TensorMap
// ============================================================

// Create a TensorMap (owned by caller)
TM_TensorMap* TM_TensorMap_Create(void);
void TM_TensorMap_Destroy(TM_TensorMap* map);

// Tensor creation and insertion into map
// For CPU tensors with data pointer (no copy)
void TM_TensorMap_SetInt32(TM_TensorMap* map, const char* name, const int32_t* data, int ndim, const int64_t* shape);
void TM_TensorMap_SetInt64(TM_TensorMap* map, const char* name, const int64_t* data, int ndim, const int64_t* shape);
void TM_TensorMap_SetFloat32(TM_TensorMap* map, const char* name, const float* data, int ndim, const int64_t* shape);
void TM_TensorMap_SetBytes(TM_TensorMap* map, const char* name, const void* data, size_t size, int ndim, const int64_t* shape);

// GPU tensor creation (data stays on GPU)
void TM_TensorMap_SetInt32GPU(TM_TensorMap* map, const char* name, const int32_t* data, int ndim, const int64_t* shape);
void TM_TensorMap_SetInt64GPU(TM_TensorMap* map, const char* name, const int64_t* data, int ndim, const int64_t* shape);
void TM_TensorMap_SetFloat32GPU(TM_TensorMap* map, const char* name, const float* data, int ndim, const int64_t* shape);

// Get tensor from map (caller does not own)
bool TM_TensorMap_Get(
    TM_TensorMap* map,
    const char* name,
    void** out_data,
    TM_DataType* out_dtype,
    TM_MemoryType* out_memory_type,
    int* out_ndim,
    int64_t* out_shape);

// ============================================================
// Generation config (sampling parameters)
// ============================================================

typedef struct TM_GenerationConfig TM_GenerationConfig;

TM_GenerationConfig* TM_GenerationConfig_Create(void);
void TM_GenerationConfig_Destroy(TM_GenerationConfig* config);

void TM_GenerationConfig_SetMaxNewTokens(TM_GenerationConfig* config, int value);
void TM_GenerationConfig_SetMinNewTokens(TM_GenerationConfig* config, int value);
void TM_GenerationConfig_SetEosIds(TM_GenerationConfig* config, const int* ids, int count);
void TM_GenerationConfig_SetStopIds(TM_GenerationConfig* config, const int* ids, int count);
void TM_GenerationConfig_SetBadIds(TM_GenerationConfig* config, const int* ids, int count);
void TM_GenerationConfig_SetTopP(TM_GenerationConfig* config, float value);
void TM_GenerationConfig_SetTopK(TM_GenerationConfig* config, int value);
void TM_GenerationConfig_SetMinP(TM_GenerationConfig* config, float value);
void TM_GenerationConfig_SetTemperature(TM_GenerationConfig* config, float value);
void TM_GenerationConfig_SetRepetitionPenalty(TM_GenerationConfig* config, float value);
void TM_GenerationConfig_SetRandomSeed(TM_GenerationConfig* config, uint64_t value);
void TM_GenerationConfig_SetOutputLogprobs(TM_GenerationConfig* config, int value);
void TM_GenerationConfig_SetOutputLastHiddenState(TM_GenerationConfig* config, int value);
void TM_GenerationConfig_SetOutputLogits(TM_GenerationConfig* config, int value);

// ============================================================
// Session parameters
// ============================================================

typedef struct {
    uint64_t id;
    int step;
    bool start_flag;
    bool end_flag;
} TM_SessionParam;

// ============================================================
// Inference (ModelRequest)
// ============================================================

// Create inference request from TurboMind instance
// Returns NULL on error
TM_ModelRequest* TM_ModelRequest_Create(TM_TurboMind* tm);
void TM_ModelRequest_Destroy(TM_ModelRequest* req);

// Forward inference
// input_tensors: owned by caller, referenced during call
// output_tensors: owned by caller, filled by function
// Returns 0 on success, negative on error
// On success, output_tensors contains:
//   - "output_ids": [batch, max_seq_len] int32, generated token IDs
//   - "sequence_length": [batch] int32, actual sequence lengths
//   - "stopping_criteria": [batch] int32, stop reason
// On error, check TM_GetLastError()
int TM_ModelRequest_Forward(
    TM_ModelRequest* req,
    TM_TensorMap* input_tensors,
    const TM_SessionParam* session,
    const TM_GenerationConfig* gen_cfg,
    bool stream_output,
    bool enable_metrics,
    TM_TensorMap* output_tensors);

// Cancel running request
void TM_ModelRequest_Cancel(TM_ModelRequest* req);

// End session (signals request completion)
void TM_ModelRequest_End(TM_ModelRequest* req, uint64_t session_id);

// Request status codes (from Request::Status enum)
typedef enum {
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
} TM_RequestStatus;

// Get request state
// Returns 0 on success, -1 if state is null
int TM_ModelRequest_GetState(TM_ModelRequest* req, TM_RequestStatus* out_status, int* out_seq_len);

// Get output tensor by name from the completed request
// Returns 0 on success, -1 if tensor not found or request not completed
int TM_ModelRequest_GetOutput(
    TM_ModelRequest* req,
    const char* name,
    void** out_data,
    size_t* out_size);

#ifdef __cplusplus
}
#endif