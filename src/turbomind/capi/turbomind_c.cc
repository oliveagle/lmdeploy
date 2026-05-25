// Copyright (c) OpenMMLab. All rights reserved.
// C API implementation for TurboMind inference engine

#include "turbomind_c.h"
#include <cuda_runtime.h>
#include <cstdio>

// DLPack header for tensor exchange protocol
#include "../python/dlpack.h"

// Compile-time debug flag for safetensors loading
// Define to 1 to enable verbose debug output during weight loading
#define SAFETENSORS_DEBUG 0

#if SAFETENSORS_DEBUG
#define SAFETENSORS_LOG(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__); fflush(stderr)
#else
#define SAFETENSORS_LOG(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#endif

// Debug function to write to a file
static void debug_log(const char* msg) {
    FILE* f = fopen("/tmp/turbomind_debug.log", "a");
    if (f) {
        fputs(msg, f);
        fclose(f);
    }
}

// Helper function to replace all occurrences of a substring
// More efficient than repeated find+replace in while loops
static void replace_all(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // Move past the replacement to avoid infinite loops
    }
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/turbomind/core/module.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/models/model_root.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/turbomind.h"
#include "src/turbomind/utils/weight_serializer.h"
#include "src/turbomind/utils/hf_config_parser.h"

namespace {

// Thread-local error state
thread_local TM_Error g_last_error = {TM_OK, {}};

constexpr TM_DataType ToCDataType(turbomind::DataType dt)
{
    using DT = turbomind::DataType;
    switch (dt) {
        case DT::kNull:      return TM_DATATYPE_INVALID;
        case DT::kBool:      return TM_DATATYPE_BOOL;
        case DT::kUint8:     return TM_DATATYPE_UINT8;
        case DT::kUint16:    return TM_DATATYPE_UINT16;
        case DT::kUint32:    return TM_DATATYPE_UINT32;
        case DT::kUint64:    return TM_DATATYPE_UINT64;
        case DT::kInt8:      return TM_DATATYPE_INT8;
        case DT::kInt16:     return TM_DATATYPE_INT16;
        case DT::kInt32:     return TM_DATATYPE_INT32;
        case DT::kInt64:     return TM_DATATYPE_INT64;
        case DT::kFloat16:   return TM_DATATYPE_FP16;
        case DT::kFloat32:   return TM_DATATYPE_FP32;
        case DT::kFloat64:   return TM_DATATYPE_FP64;
        case DT::kBfloat16:  return TM_DATATYPE_BF16;
        case DT::kFloat8_e4m3: return TM_DATATYPE_FP8_E4M3;
        case DT::kFloat4_e2m1: return TM_DATATYPE_FP4_E2M1;
        case DT::kUint4:     return TM_DATATYPE_UINT4;
        default:             return TM_DATATYPE_INVALID;
    }
}

constexpr turbomind::DataType FromCDataType(TM_DataType dt)
{
    using DT = turbomind::DataType;
    switch (dt) {
        case TM_DATATYPE_BOOL:         return DT::kBool;
        case TM_DATATYPE_UINT8:        return DT::kUint8;
        case TM_DATATYPE_UINT16:       return DT::kUint16;
        case TM_DATATYPE_UINT32:       return DT::kUint32;
        case TM_DATATYPE_UINT64:       return DT::kUint64;
        case TM_DATATYPE_INT8:         return DT::kInt8;
        case TM_DATATYPE_INT16:        return DT::kInt16;
        case TM_DATATYPE_INT32:        return DT::kInt32;
        case TM_DATATYPE_INT64:        return DT::kInt64;
        case TM_DATATYPE_FP16:          return DT::kFloat16;
        case TM_DATATYPE_FP32:         return DT::kFloat32;
        case TM_DATATYPE_FP64:         return DT::kFloat64;
        case TM_DATATYPE_BF16:         return DT::kBfloat16;
        case TM_DATATYPE_FP8_E4M3:     return DT::kFloat8_e4m3;
        case TM_DATATYPE_FP4_E2M1:     return DT::kFloat4_e2m1;
        case TM_DATATYPE_UINT4:        return DT::kUint4;
        default:                       return DT::kNull;
    }
}

constexpr TM_MemoryType ToCMemoryType(turbomind::DeviceType mt)
{
    switch (mt) {
        case turbomind::DeviceType::kCPU:        return TM_MEMORY_CPU;
        case turbomind::DeviceType::kCPUpinned:  return TM_MEMORY_CPU_PINNED;
        case turbomind::DeviceType::kDEVICE:     return TM_MEMORY_GPU;
        default:                                 return TM_MEMORY_CPU;
    }
}

turbomind::DeviceType FromCMemoryType(TM_MemoryType mt)
{
    switch (mt) {
        case TM_MEMORY_CPU:          return turbomind::DeviceType::kCPU;
        case TM_MEMORY_CPU_PINNED:    return turbomind::DeviceType::kCPUpinned;
        case TM_MEMORY_GPU:          return turbomind::DeviceType::kDEVICE;
        default:                     return turbomind::DeviceType::kCPU;
    }
}

void SetError(TM_ErrorCode code, const char* msg)
{
    g_last_error.code = code;
    strncpy(g_last_error.message, msg, sizeof(g_last_error.message) - 1);
    g_last_error.message[sizeof(g_last_error.message) - 1] = '\0';
}

}  // anonymous namespace

// ============================================================
// Error handling
// ============================================================

TM_Error* TM_GetLastError(void)
{
    if (g_last_error.code == TM_OK) {
        return nullptr;
    }
    return &g_last_error;
}

void TM_ClearError(void)
{
    g_last_error.code = TM_OK;
    g_last_error.message[0] = '\0';
}

// ============================================================
// Engine configuration
// ============================================================

struct TM_EngineConfig {
    turbomind::EngineConfig config;
    std::vector<int> devices;
};

TM_EngineConfig* TM_EngineConfig_Create(void)
{
    try {
        auto* cfg = new TM_EngineConfig{};
        cfg->config.enable_prefix_caching = false;
        cfg->config.enable_metrics = false;
        // data_type defaults to kHalf via EngineConfig default; callers may
        // override via TM_EngineConfig_SetDataType() (e.g. BF16 on supported hardware).
        return cfg;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

void TM_EngineConfig_Destroy(TM_EngineConfig* config)
{
    delete config;
}

void TM_EngineConfig_SetDataType(TM_EngineConfig* config, TM_DataType data_type)
{
    config->config.data_type = FromCDataType(data_type);
}

void TM_EngineConfig_SetCacheBlockSeqLen(TM_EngineConfig* config, int value)
{
    config->config.cache_block_seq_len = value;
}

void TM_EngineConfig_SetQuantPolicy(TM_EngineConfig* config, int value)
{
    config->config.quant_policy = value;
}

void TM_EngineConfig_SetTuneLayerNum(TM_EngineConfig* config, int value)
{
    config->config.tune_layer_num = value;
}

void TM_EngineConfig_SetMaxBatchSize(TM_EngineConfig* config, int value)
{
    config->config.max_batch_size = value;
}

void TM_EngineConfig_SetMaxPrefillTokenNum(TM_EngineConfig* config, int value)
{
    config->config.max_prefill_token_num = value;
}

void TM_EngineConfig_SetMaxContextTokenNum(TM_EngineConfig* config, int value)
{
    config->config.max_context_token_num = value;
}

void TM_EngineConfig_SetSessionLen(TM_EngineConfig* config, int value)
{
    config->config.session_len = value;
}

void TM_EngineConfig_SetCacheMaxBlockCount(TM_EngineConfig* config, float value)
{
    config->config.cache_max_block_count = value;
}

void TM_EngineConfig_SetCacheChunkSize(TM_EngineConfig* config, int value)
{
    config->config.cache_chunk_size = value;
}

void TM_EngineConfig_SetEnablePrefixCaching(TM_EngineConfig* config, bool value)
{
    config->config.enable_prefix_caching = value;
}

void TM_EngineConfig_SetEnableMetrics(TM_EngineConfig* config, bool value)
{
    config->config.enable_metrics = value;
}

void TM_EngineConfig_SetNumTokensPerIter(TM_EngineConfig* config, int value)
{
    config->config.num_tokens_per_iter = value;
}

void TM_EngineConfig_SetMaxPrefillIters(TM_EngineConfig* config, int value)
{
    config->config.max_prefill_iters = value;
}

void TM_EngineConfig_SetAsync(TM_EngineConfig* config, int value)
{
    config->config.async_ = value;
}

void TM_EngineConfig_SetOuterDpSize(TM_EngineConfig* config, int value)
{
    config->config.outer_dp_size = value;
}

void TM_EngineConfig_SetAttnDpSize(TM_EngineConfig* config, int value)
{
    config->config.attn_dp_size = value;
}

void TM_EngineConfig_SetAttnTpSize(TM_EngineConfig* config, int value)
{
    config->config.attn_tp_size = value;
}

void TM_EngineConfig_SetAttnCpSize(TM_EngineConfig* config, int value)
{
    config->config.attn_cp_size = value;
}

void TM_EngineConfig_SetMlpTpSize(TM_EngineConfig* config, int value)
{
    config->config.mlp_tp_size = value;
}

void TM_EngineConfig_SetNNodes(TM_EngineConfig* config, int value)
{
    config->config.nnodes = value;
}

void TM_EngineConfig_SetNodeRank(TM_EngineConfig* config, int value)
{
    config->config.node_rank = value;
}

void TM_EngineConfig_SetCommunicator(TM_EngineConfig* config, const char* value)
{
    if (value) {
        config->config.communicator = value;
    }
}

void TM_EngineConfig_AddDevice(TM_EngineConfig* config, int device_id)
{
    config->devices.push_back(device_id);
    config->config.devices.push_back(device_id);
}

// ============================================================
// TurboMind instance
// ============================================================

struct TM_TurboMind {
    // No-FFI factory: TurboMind normally requires a GIL factory for async operations.
    // We provide a simple one here.
    std::unique_ptr<turbomind::TurboMind> instance;
};

TM_TurboMind* TM_TurboMind_Create(const char* model_dir, TM_EngineConfig* config)
{
    if (!model_dir || !config) {
        SetError(TM_ERR_INVALID_ARG, "model_dir and config must not be NULL");
        return nullptr;
    }
    if (config->devices.empty()) {
        SetError(TM_ERR_INVALID_ARG, "no devices configured");
        return nullptr;
    }

    try {
        fprintf(stderr, "[C-API] TM_TurboMind_Create: config.data_type=%d (expect kHalf=%d or kBfloat16=%d)\n",
                (int)config->config.data_type, (int)turbomind::DataType::kHalf, (int)turbomind::DataType::kBfloat16);
        fflush(stderr);

        auto* tm = new TM_TurboMind{};

        // Create a simple GIL factory (no-op for C API)
        auto gil_factory = []() -> std::shared_ptr<void> { return nullptr; };

        tm->instance = std::make_unique<turbomind::TurboMind>(
            model_dir,
            std::move(config->config),
            std::move(gil_factory));

        fprintf(stderr, "[C-API] TM_TurboMind_Create: instance created successfully\n");
        fflush(stderr);

        return tm;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

void TM_TurboMind_Destroy(TM_TurboMind* tm)
{
    delete tm;
}

void TM_TurboMind_CreateContext(TM_TurboMind* tm, int index)
{
    tm->instance->CreateContext(index);
}

void TM_TurboMind_CreateRoot(TM_TurboMind* tm, int index)
{
    tm->instance->CreateRoot(index);
}

void TM_TurboMind_ProcessWeights(TM_TurboMind* tm, int index)
{
    tm->instance->ProcessWeights(index);
}

void TM_TurboMind_CreateEngine(TM_TurboMind* tm, int index)
{
    tm->instance->CreateEngine(index);
}

bool TM_TurboMind_IsDummyNode(TM_TurboMind* tm)
{
    return tm->instance->is_dummy_node();
}

int TM_TurboMind_GetAttnTpRank(TM_TurboMind* tm, int index)
{
    return tm->instance->GetAttnTpRank(index);
}

int TM_TurboMind_GetMlpTpRank(TM_TurboMind* tm, int index)
{
    return tm->instance->GetMlpTpRank(index);
}

int TM_TurboMind_GetModelTpRank(TM_TurboMind* tm, int index)
{
    return tm->instance->GetModelTpRank(index);
}

// ============================================================
// High-level initialization
// ============================================================

namespace {

// HuggingFace config.json structure
struct HfModelConfig {
    // Basic model parameters
    int hidden_size = 4096;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = -1;  // -1 means same as num_attention_heads
    int vocab_size = 32000;
    int intermediate_size = 0;  // Will default to hidden_size * 4
    int max_position_embeddings = 8192;

    // Model architecture
    std::string model_type = "llama";
    std::string arch = "Transformer";

    // MoE configuration
    int num_local_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;

    // MTP (Multi-Token Prediction) configuration
    // Qwen3.5: mtp_num_hidden_layers (mtp.layers.* weights)
    // DeepSeek/GLM4: num_nextn_predict_layers (layers.{num_hidden_layers+N}.* weights)
    int mtp_num_hidden_layers = 0;
    int num_nextn_predict_layers = 0;

    // DeltaNet / Linear Attention
    bool use_linear_attn = false;

    // RoPE configuration
    int rope_dim = 0;
    float rope_scaling_factor = 1.0f;

    // Quantization
    bool is_awq = false;
    int awq_bits = 4;
    int awq_group_size = 128;
    std::string awq_version = "gemm";

    // Nested config support (Qwen3.5 MoE, multimodal models)
    bool has_text_config = false;
    bool has_model_config = false;

    // Per-layer attention types (Qwen3.5)
    // If empty, fall back to use_linear_attn for all layers
    std::vector<bool> layer_is_linear_attn;
};

// Parse HuggingFace config.json using the HfConfigParser
static HfModelConfig ParseHfConfig(const std::string& model_dir)
{
    HfModelConfig config;

    std::string config_path = model_dir;
    if (!config_path.empty() && config_path.back() != '/') {
        config_path += '/';
    }
    config_path += "config.json";

    auto root = turbomind::HfConfigParser::ParseFile(config_path);
    if (root.is_null()) {
        return config;  // Return defaults on parse failure
    }

    // Determine config source (handle nested configs)
    const auto& config_source = [&]() -> const turbomind::HfConfigParser::Value& {
        if (root.has("text_config")) {
            config.has_text_config = true;
            return root.get("text_config");
        } else if (root.has("model_config")) {
            config.has_model_config = true;
            return root.get("model_config");
        }
        return root;
    }();

    // Helper to get int value with fallback
    auto get_int = [&](const std::string& key, int default_val) -> int {
        auto val = config_source.get(key);
        if (val.is_int()) return static_cast<int>(val.as_int());
        if (val.is_float()) return static_cast<int>(val.as_float());
        return default_val;
    };

    // Helper to get string value
    auto get_string = [&](const std::string& key, const std::string& default_val) -> std::string {
        auto val = config_source.get(key);
        return val.is_string() ? val.as_string() : default_val;
    };

    // Helper to get bool value
    auto get_bool = [&](const std::string& key, bool default_val) -> bool {
        auto val = config_source.get(key);
        return val.is_bool() ? val.as_bool() : default_val;
    };

    // Parse basic model parameters
    config.hidden_size = get_int("hidden_size", config.hidden_size);
    config.num_hidden_layers = get_int("num_hidden_layers", config.num_hidden_layers);
    config.num_attention_heads = get_int("num_attention_heads", config.num_attention_heads);
    config.num_key_value_heads = get_int("num_key_value_heads", config.num_attention_heads);
    config.vocab_size = get_int("vocab_size", config.vocab_size);
    config.max_position_embeddings = get_int("max_position_embeddings", config.max_position_embeddings);

    // Intermediate size (default to 4x hidden_size if not specified)
    // For MoE models, check moe_intermediate_size first (Qwen3.5 MoE)
    if (get_int("moe_intermediate_size", 0) > 0) {
        // MoE model: use moe_intermediate_size for expert FFNs
        config.intermediate_size = get_int("moe_intermediate_size", config.hidden_size * 4);
    } else {
        config.intermediate_size = get_int("intermediate_size", config.hidden_size * 4);
    }

    // Model type
    config.model_type = get_string("model_type", "llama");
    config.arch = get_string("arch", config.arch);

    // MoE configuration
    // Qwen3.5: "num_experts" (in text_config); others: "num_local_experts"
    config.num_local_experts = get_int("num_local_experts", 0);
    if (config.num_local_experts == 0) {
        config.num_local_experts = get_int("num_experts", 0);
    }
    // num_experts_per_tok (same name across models)
    config.num_experts_per_tok = get_int("num_experts_per_tok", 0);
    // moe_intermediate_size (Qwen3.5) vs intermediate_size (fallback for other models)
    config.moe_intermediate_size = get_int("moe_intermediate_size", 0);
    if (config.moe_intermediate_size == 0) {
        config.moe_intermediate_size = get_int("intermediate_size", 0);
    }

    // MTP (Multi-Token Prediction) configuration
    // Qwen3.5: mtp_num_hidden_layers (mtp.layers.* weights)
    // DeepSeek/GLM4: num_nextn_predict_layers (layers.{num_hidden_layers+N}.* weights)
    config.mtp_num_hidden_layers = get_int("mtp_num_hidden_layers", 0);
    config.num_nextn_predict_layers = get_int("num_nextn_predict_layers", 0);

    // DeltaNet / Linear Attention
    // Qwen3.5 uses layer_types array (["linear_attention", "full_attention", ...])
    // Older models use use_linear_attn boolean
    config.use_linear_attn = get_bool("use_linear_attn", false);

    // Parse layer_types if present (Qwen3.5)
    const auto& layer_types_val = config_source.get("layer_types");
    if (!layer_types_val.is_null() && layer_types_val.is_array()) {
        const auto& layer_types_arr = layer_types_val.as_array();
        config.layer_is_linear_attn.reserve(layer_types_arr.size());
        for (const auto& layer_type : layer_types_arr) {
            if (layer_type.is_string()) {
                const std::string type_str = layer_type.as_string();
                // "linear_attention" means use DeltaNetWeight
                // "full_attention" means use AttentionWeight
                config.layer_is_linear_attn.push_back(type_str == "linear_attention");
            } else {
                config.layer_is_linear_attn.push_back(false);
            }
        }
    } else {
        // No layer_types specified, use use_linear_attn for all layers
        config.layer_is_linear_attn.resize(config.num_hidden_layers, config.use_linear_attn);
    }

    // RoPE configuration
    config.rope_dim = get_int("rope_dim", 0);
    config.rope_scaling_factor = static_cast<float>(
        config_source.get("rope_scaling").get("factor").as_float(1.0)
    );

    // Parse quantization_config for AWQ
    const auto& quant_config = root.get("quantization_config");
    if (!quant_config.is_null()) {
        std::string quant_method = quant_config.get("quant_method").as_string("");
        if (quant_method == "awq") {
            config.is_awq = true;
            config.awq_bits = static_cast<int>(quant_config.get("bits").as_int(4));
            config.awq_group_size = static_cast<int>(quant_config.get("group_size").as_int(128));
            config.awq_version = quant_config.get("version").as_string("gemm");
        }
    }

    return config;
}

}  // anonymous namespace

// ============================================================
// Safetensors file handling (using header-only implementation)
// ============================================================

#include "src/turbomind/utils/safetensors_reader.h"
#include "src/turbomind/utils/safetensors_reader_mmap.h"

// Wrapper to adapt turbomind::SafetensorsReader to the C API
// The C API uses an opaque void* handle, so we wrap the C++ reader

namespace {

// Map turbomind::DataType to TM_DataType (C API)
static TM_DataType ToCApiDtype(turbomind::DataType dt)
{
    switch (dt) {
        case turbomind::DataType::kNull:        return TM_DATATYPE_INVALID;
        case turbomind::DataType::kBool:        return TM_DATATYPE_BOOL;
        case turbomind::DataType::kUint8:       return TM_DATATYPE_UINT8;
        case turbomind::DataType::kUint16:      return TM_DATATYPE_UINT16;
        case turbomind::DataType::kUint32:      return TM_DATATYPE_UINT32;
        case turbomind::DataType::kUint64:      return TM_DATATYPE_UINT64;
        case turbomind::DataType::kInt8:        return TM_DATATYPE_INT8;
        case turbomind::DataType::kInt16:       return TM_DATATYPE_INT16;
        case turbomind::DataType::kInt32:       return TM_DATATYPE_INT32;
        case turbomind::DataType::kInt64:       return TM_DATATYPE_INT64;
        case turbomind::DataType::kFloat16:     return TM_DATATYPE_FP16;
        case turbomind::DataType::kFloat32:     return TM_DATATYPE_FP32;
        case turbomind::DataType::kFloat64:     return TM_DATATYPE_FP64;
        case turbomind::DataType::kBfloat16:    return TM_DATATYPE_BF16;
        case turbomind::DataType::kFloat8_e4m3: return TM_DATATYPE_FP8_E4M3;
        case turbomind::DataType::kFloat4_e2m1: return TM_DATATYPE_FP4_E2M1;
        case turbomind::DataType::kUint4:       return TM_DATATYPE_UINT4;
        default:                                return TM_DATATYPE_INVALID;
    }
}

/// Create a LinearConfig appropriate for AWQ 4-bit quantized weights.
/// When AWQ is enabled, weight dtype is kUint4 (4-bit packed), scales/zeros use model activation dtype.
/// For non-AWQ, uses plain row-major format with activation dtype.
static turbomind::core::LinearConfig CreateAwqLinearConfig(
    int input_dim,
    int output_dim,
    turbomind::DataType data_type,
    bool is_awq,
    int awq_group_size)
{
    turbomind::core::LinearConfig cfg;
    cfg.input_dim = input_dim;
    cfg.output_dim = output_dim;
    cfg.data_type = data_type;
    cfg.has_bias = false;

    if (is_awq) {
        cfg.format = turbomind::ResolveLinearWeightFormat(data_type, turbomind::kUint4, awq_group_size, 1);
    } else {
        cfg.format = turbomind::DataFormat{};
    }

    return cfg;
}

}  // anonymous namespace

void* TM_Safetensors_Open(const char* file_path)
{
    try {
        // Return a pointer to a newly created SafetensorsReader
        // Caller is responsible for calling TM_Safetensors_Close to delete
        return new turbomind::SafetensorsReader(file_path);
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

void TM_Safetensors_Close(void* handle)
{
    if (handle) {
        delete static_cast<turbomind::SafetensorsReader*>(handle);
    }
}

int TM_Safetensors_GetTensor(
    void* handle,
    const char* name,
    void** out_data,
    size_t* out_size,
    int* out_ndim,
    int64_t* out_shape,
    TM_DataType* out_dtype)
{
    if (!handle || !name || !out_data || !out_size || !out_ndim || !out_shape || !out_dtype) {
        return TM_ERR_INVALID_ARG;
    }

    auto* reader = static_cast<turbomind::SafetensorsReader*>(handle);

    // Get tensor metadata
    const auto* meta = reader->get_tensor_meta(name);
    if (!meta) {
        SetError(TM_ERR_NOT_FOUND, "Tensor not found");
        return -1;
    }

    // Allocate buffer and read tensor data
    std::vector<uint8_t> data = reader->read_tensor(name);
    if (data.empty()) {
        SetError(TM_ERR_RUNTIME, "Failed to read tensor data");
        return -1;
    }

    // Copy data to caller-owned buffer
    char* data_copy = new char[data.size()];
    std::memcpy(data_copy, data.data(), data.size());
    *out_data = data_copy;
    *out_size = data.size();

    // Return shape info
    *out_ndim = static_cast<int>(meta->shape.size());
    for (size_t i = 0; i < meta->shape.size(); ++i) {
        out_shape[i] = static_cast<int64_t>(meta->shape[i]);
    }
    *out_dtype = ToCApiDtype(meta->dtype);

    return 0;
}

int TM_Safetensors_NumTensors(void* handle)
{
    if (!handle) return 0;
    auto* reader = static_cast<turbomind::SafetensorsReader*>(handle);
    return static_cast<int>(reader->num_tensors());
}

const char* TM_Safetensors_GetTensorName(void* handle, int index)
{
    if (!handle) return nullptr;
    auto* reader = static_cast<turbomind::SafetensorsReader*>(handle);
    if (index < 0 || index >= static_cast<int>(reader->num_tensors())) return nullptr;
    // Note: returned pointer is only valid until SafetensorsReader is destroyed
    static thread_local std::string last_name;
    last_name = reader->tensor_name(static_cast<size_t>(index));
    return last_name.c_str();
}

// ============================================================
// Helper functions for weight loading
// ============================================================

// Forward declarations
static std::string MapHuggingFaceWeightToTurboMind(const std::string& hf_name, const HfModelConfig& hf_config);
static void LoadWeightsFromSafetensors(
    turbomind::ModelWeight* model_weight,
    const char* safetensors_path,
    const HfModelConfig& hf_config);

// Forward declarations for debugging
namespace turbomind {
core::Param LinearWeight_debug_param(class LinearWeight* lw, const std::string& name);
}

// Read an integer value from config.json (kept for backward compatibility with non-critical reads)
static int ReadIntFromConfig(const std::string& model_dir, const std::string& key, int default_val)
{
    std::string config_path = model_dir;
    if (!config_path.empty() && config_path.back() != '/') {
        config_path += '/';
    }
    config_path += "config.json";

    auto root = turbomind::HfConfigParser::ParseFile(config_path);
    if (root.is_null()) return default_val;

    auto val = root.get(key);
    if (val.is_int()) return static_cast<int>(val.as_int());
    if (val.is_float()) return static_cast<int>(val.as_float());
    return default_val;
}

// Find all safetensors files in the model directory
static std::vector<std::string> FindSafetensorsFiles(const std::string& model_dir)
{
    std::vector<std::string> files;

    // First, try to read model.safetensors.index.json
    std::string index_path = model_dir;
    if (!index_path.empty() && index_path.back() != '/') {
        index_path += '/';
    }
    index_path += "model.safetensors.index.json";

    std::ifstream index_file(index_path);
    if (index_file.is_open()) {
        // Parse the index file to get weight map
        std::string content((std::istreambuf_iterator<char>(index_file)),
                           std::istreambuf_iterator<char>());

        // Look for "weight_map": { ... }
        size_t map_start = content.find("\"weight_map\":");
        if (map_start != std::string::npos) {
            // Find all filename references
            size_t pos = map_start;
            while ((pos = content.find("\"", pos)) != std::string::npos) {
                size_t start = pos + 1;
                size_t end = content.find("\"", start);
                if (end == std::string::npos) break;
                std::string value = content.substr(start, end - start);

                // If it looks like a filename (contains .safetensors), add it
                if (value.find(".safetensors") != std::string::npos) {
                    std::string full_path = model_dir;
                    if (!full_path.empty() && full_path.back() != '/') {
                        full_path += '/';
                    }
                    full_path += value;

                    // Check if already in list
                    if (std::find(files.begin(), files.end(), full_path) == files.end()) {
                        files.push_back(full_path);
                    }
                }
                pos = end + 1;
            }
        }
    }

    // Fallback: glob for .safetensors files
    if (files.empty()) {
        // Try common patterns
        std::vector<std::string> patterns = {
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00001-of-00003.safetensors",
        };

        for (const auto& pattern : patterns) {
            std::string full_path = model_dir;
            if (!full_path.empty() && full_path.back() != '/') {
                full_path += '/';
            }
            full_path += pattern;

            std::ifstream test_file(full_path);
            if (test_file.is_open()) {
                files.push_back(full_path);

                // For sharded files, try to find all shards
                if (pattern.find("-of-") != std::string::npos) {
                    // Extract shard count
                    size_t of_pos = pattern.find("-of-");
                    size_t shard_start = of_pos + 4;
                    size_t shard_end = pattern.find(".safetensors", shard_start);
                    if (shard_end != std::string::npos) {
                        int num_shards = std::stoi(pattern.substr(shard_start, shard_end - shard_start));
                        for (int i = 2; i <= num_shards; ++i) {
                            std::string shard_pattern = pattern;
                            std::string shard_num = std::to_string(i);
                            if (shard_num.length() == 1) {
                                shard_num = "0" + shard_num;
                            }
                            shard_pattern.replace(of_pos + 1, 5, shard_num + "-of-" + std::to_string(num_shards));

                            std::string shard_path = model_dir;
                            if (!shard_path.empty() && shard_path.back() != '/') {
                                shard_path += '/';
                            }
                            shard_path += shard_pattern;

                            std::ifstream shard_file(shard_path);
                            if (shard_file.is_open()) {
                                files.push_back(shard_path);
                            }
                        }
                    }
                }
                break;
            }
        }
    }

    return files;
}

// Load weights from a safetensors file and populate the ModelWeight module
// Optimized version: batches cudaMemcpyAsync calls for better performance
static void LoadWeightsFromSafetensors(
    turbomind::ModelWeight* model_weight,
    const char* safetensors_path,
    const HfModelConfig& hf_config)
{
    // Push a CUDA managed memory allocator to allow weight loading without CUDA OOM
    // cudaMallocManaged uses unified memory that can be oversubscribed on systems
    // with more system RAM than GPU VRAM
    turbomind::core::Allocator managed_allocator{turbomind::core::CreateCudaManagedAllocator()};
    auto managed_guard = turbomind::core::ContextGuard{managed_allocator};

    SAFETENSORS_LOG("[C-API] LoadWeightsFromSafetensors: %s\n", safetensors_path);

    try {
        auto load_start = std::chrono::high_resolution_clock::now();

        // Use mmap-based reader for zero-copy tensor access
        turbomind::SafetensorsReaderMmap reader(safetensors_path);

        auto mmap_done = std::chrono::high_resolution_clock::now();
        auto mmap_ms = std::chrono::duration_cast<std::chrono::milliseconds>(mmap_done - load_start);
        SAFETENSORS_LOG("[C-API] mmap took: %ldms, Loading safetensors: %s (%zu tensors)\n", mmap_ms.count(), safetensors_path, reader.num_tensors());

        // Pre-allocated vectors (reused for all tensors to avoid heap allocations)
        std::vector<std::string> parts;
        parts.reserve(8);  // Typical path: layers.X.attention.w1.weight = 5 parts

        std::vector<size_t> shape_vec;
        shape_vec.reserve(4);  // Typical tensor rank <= 4

        int loaded_count = 0;
        int skip_count = 0;

        // Set CUDA device before creating stream
        cudaSetDevice(0);

        // Create BatchCopy for batched transfers
        turbomind::core::BatchCopy batch_copy;

        // Phase 1: Iterate tensors, map names, allocate memory, and collect transfers
        // We'll collect all transfers first, then run them in a single batch
        size_t progress_count = 0;
        const auto& tensors = reader.tensors();  // Direct reference to avoid re-lookup

        SAFETENSORS_LOG("[C-API] Starting tensor processing loop (%zu tensors)...\n", reader.num_tensors());

        // Structure to hold transfer info
        struct Transfer {
            const uint8_t* src;
            void* dst;
            size_t size;
        };
        std::vector<Transfer> transfers;
        transfers.reserve(reader.num_tensors());

        // Temporary storage for fused QKV loading
        // HF stores separate q_proj/k_proj/v_proj, but TM uses fused w_qkv
        struct QKVAccumulator {
            std::vector<uint8_t> q_data;
            std::vector<uint8_t> k_data;
            std::vector<uint8_t> v_data;
            std::vector<size_t> q_shape;
            std::vector<size_t> k_shape;
            std::vector<size_t> v_shape;
            turbomind::DataType dtype = turbomind::DataType::kNull;
            bool q_loaded = false;
            bool k_loaded = false;
            bool v_loaded = false;
        };
        std::unordered_map<std::string, QKVAccumulator> qkv_accumulators;

        for (size_t i = 0; i < reader.num_tensors(); ++i) {
            const auto& meta = tensors[i];  // Direct reference, no extra lookup
            const std::string& tensor_name = meta.name;

            // Map HF weight names to TurboMind module paths
            std::string tm_path = MapHuggingFaceWeightToTurboMind(tensor_name, hf_config);
            if (tm_path.empty()) {
                ++skip_count;
                continue;
            }

            // Check if this is a QKV projection that needs to be fused
            // Detect pattern: .w_qkv. suffix (after our mapping from q/k/v_proj)
            bool is_qkv_proj = (tm_path.find(".w_qkv.") != std::string::npos);
            std::string qkv_key;  // e.g., "layers.0.attention" for qkv accumulator

            if (is_qkv_proj) {
                // Extract the attention module path as key
                // tm_path like "layers.0.attention.w_qkv.weight" -> "layers.0.attention"
                size_t last_dot = tm_path.rfind('.');
                if (last_dot != std::string::npos) {
                    size_t second_last_dot = tm_path.rfind('.', last_dot - 1);
                    if (second_last_dot != std::string::npos) {
                        qkv_key = tm_path.substr(0, second_last_dot);
                    }
                }
            }

            // Parse the TurboMind path to find the module and param
            // Fast manual parsing (avoids stringstream overhead)
            parts.clear();
            size_t start = 0;
            size_t dot_pos;
            while ((dot_pos = tm_path.find('.', start)) != std::string::npos) {
                parts.push_back(tm_path.substr(start, dot_pos - start));
                start = dot_pos + 1;
            }
            parts.push_back(tm_path.substr(start));

            if (parts.empty()) {
                continue;
            }

            // Navigate to the target module
            turbomind::core::Module* current = model_weight;
            if (parts.size() > 1) {
                for (size_t j = 0; j < parts.size() - 1; ++j) {
                    if (!current) break;
                    current = current->child(parts[j]);
                }
            }

            if (!current) {
                ++skip_count;
                continue;
            }

            // Use param() for O(1) lookup instead of for_each_param() iteration
            const std::string& param_name = parts.back();
            turbomind::core::Param target_param;

            // Handle nested param names like "w1.weight" (child.param pattern)
            // This is needed for MoE experts: layers.0.moe_ffn.experts.0.w1.weight
            // where "w1.weight" means navigate to child "w1" first, then get param "weight"
            size_t param_dot_pos = param_name.find('.');
            if (param_dot_pos != std::string::npos) {
                // Split into child_name and actual param name
                std::string child_name = param_name.substr(0, param_dot_pos);
                std::string actual_param = param_name.substr(param_dot_pos + 1);
                auto* child_module = current->child(child_name);
                if (child_module) {
                    target_param = child_module->param(actual_param);
                } else {
                    SAFETENSORS_LOG("[C-API] WARNING: Child module '%s' not found for param '%s' (path: %s)\n",
                                   child_name.c_str(), param_name.c_str(), tm_path.c_str());
                    ++skip_count;
                    continue;
                }
            } else {
                target_param = current->param(param_name);
            }

            // For QKV projections, accumulate instead of direct load
            // They'll be fused later
            if (is_qkv_proj && !qkv_key.empty()) {
                auto& acc = qkv_accumulators[qkv_key];
                std::string proj_type = param_name;  // "w_qkv.weight"
                // Determine which projection this is based on original tensor name
                // We need to check the original HF tensor name before mapping
                // The mapped path still has "w_qkv" for all 3, so we track by original name

                // Get original HF tensor name to determine Q/K/V type
                // Original names are like "model.layers.0.self_attn.q_proj.weight"
                bool is_q = (tensor_name.find(".q_proj.") != std::string::npos);
                bool is_k = (tensor_name.find(".k_proj.") != std::string::npos);
                bool is_v = (tensor_name.find(".v_proj.") != std::string::npos);

                if (is_q || is_k || is_v) {
                    // Read tensor data from safetensors (zero-copy from mmap)
                    auto read_start = std::chrono::high_resolution_clock::now();
                    std::vector<uint8_t> tensor_data = reader.read_tensor(tensor_name);
                    auto read_elapsed = std::chrono::high_resolution_clock::now() - read_start;
                    size_t data_size_mb = tensor_data.size() / (1024 * 1024);
                    if (tensor_data.size() > 10 * 1024 * 1024 || read_elapsed > std::chrono::milliseconds(100)) {
                        SAFETENSORS_LOG("[C-API] Read %s tensor: %zu MB, %.1fms\n",
                                       is_q ? "Q" : (is_k ? "K" : "V"), data_size_mb,
                                       std::chrono::duration_cast<std::chrono::milliseconds>(read_elapsed).count());
                    }

                    if (is_q && acc.q_data.empty()) {
                        acc.q_data = std::move(tensor_data);
                        acc.q_shape = meta.shape;
                        acc.dtype = meta.dtype;
                        acc.q_loaded = true;
                    } else if (is_k && acc.k_data.empty()) {
                        acc.k_data = std::move(tensor_data);
                        acc.k_shape = meta.shape;
                        acc.dtype = meta.dtype;
                        acc.k_loaded = true;
                    } else if (is_v && acc.v_data.empty()) {
                        acc.v_data = std::move(tensor_data);
                        acc.v_shape = meta.shape;
                        acc.dtype = meta.dtype;
                        acc.v_loaded = true;
                    }
                    ++loaded_count;
                    continue;
                }
            }

            if (!target_param) {
                ++skip_count;
                continue;
            }

            // Get direct zero-copy pointer to mmap'd data (no CPU copy!)
            // CRITICAL: get_tensor_data can fail on some systems, check it
            const uint8_t* src_data = nullptr;
            try {
                src_data = reader.get_tensor_data(tensor_name);
            } catch (const std::exception& e) {
                SAFETENSORS_LOG("[C-API] get_tensor_data failed for %s: %s\n", tensor_name.c_str(), e.what());
                ++skip_count;
                continue;
            }

            if (!src_data) {
                SAFETENSORS_LOG("[C-API] NULL data pointer for %s\n", tensor_name.c_str());
                ++skip_count;
                continue;
            }

            // Build shape vector (reuse pre-allocated vector)
            shape_vec.clear();
            for (auto s : meta.shape) {
                shape_vec.push_back(static_cast<size_t>(s));
            }

            // Allocate GPU memory and transfer directly from mmap'd region
            auto alloc_start = std::chrono::high_resolution_clock::now();
            auto tensor = target_param.alloc(shape_vec, meta.dtype);
            auto alloc_elapsed = std::chrono::high_resolution_clock::now() - alloc_start;

            if (!tensor) {
                ++skip_count;
                continue;
            }

            if (tensor.raw_data()) {
                size_t copy_size = std::min(meta.size, static_cast<size_t>(tensor.byte_size()));
                transfers.push_back({src_data, tensor.raw_data(), copy_size});
                ++loaded_count;
            }

            ++progress_count;
            if (progress_count % 200 == 0 || i == reader.num_tensors() - 1) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - load_start);
                SAFETENSORS_LOG("[C-API] Allocated %zu/%zu tensors (%.0fs elapsed, alloc took %.0fms) from %s\n",
                               progress_count, reader.num_tensors(), elapsed.count() / 1000.0,
                               alloc_elapsed.count() / 1000.0, safetensors_path);
            }
        }

        // Phase 1.5: Fuse accumulated QKV tensors and add to transfers
        if (!qkv_accumulators.empty()) {
            SAFETENSORS_LOG("[C-API] Fusing %zu QKV tensors...\n", qkv_accumulators.size());

            for (const auto& [key, acc] : qkv_accumulators) {
                if (!acc.q_loaded || !acc.k_loaded || !acc.v_loaded) {
                    fprintf(stderr, "[C-API] WARNING: Incomplete QKV for %s (q=%d,k=%d,v=%d)\n",
                            key.c_str(), acc.q_loaded, acc.k_loaded, acc.v_loaded);
                    continue;
                }

                auto qkv_start = std::chrono::high_resolution_clock::now();

                // Navigate to the attention module
                turbomind::core::Module* attn_module = model_weight;
                parts.clear();
                size_t start = 0;
                size_t dot_pos;
                while ((dot_pos = key.find('.', start)) != std::string::npos) {
                    parts.push_back(key.substr(start, dot_pos - start));
                    start = dot_pos + 1;
                }
                parts.push_back(key.substr(start));

                for (size_t j = 0; j < parts.size() && attn_module; ++j) {
                    attn_module = attn_module->child(parts[j]);
                }

                if (!attn_module) {
                    fprintf(stderr, "[C-API] ERROR: Cannot find attn module for QKV fusion: %s\n", key.c_str());
                    continue;
                }

                // Allocate w_qkv tensor
                turbomind::core::Param w_qkv_param = attn_module->param("w_qkv.weight");
                if (!w_qkv_param) {
                    fprintf(stderr, "[C-API] ERROR: Cannot find w_qkv.weight param for %s\n", key.c_str());
                    continue;
                }

                // Calculate fused shape
                // HF: q=[hidden, q_out], k=[hidden, k_out], v=[hidden, v_out]
                // TM: w_qkv=[q_out + k_out + v_out, hidden]
                const size_t hidden = acc.q_shape[0];
                const size_t q_out = acc.q_shape[1];
                const size_t k_out = acc.k_shape[1];
                const size_t v_out = acc.v_shape[1];
                const size_t fused_out = q_out + k_out + v_out;

                // Data size calculation
                const size_t elem_size = turbomind::byte_size(acc.dtype);
                const size_t q_bytes = acc.q_data.size();
                const size_t k_bytes = acc.k_data.size();
                const size_t v_bytes = acc.v_data.size();

                // Allocate fused tensor on GPU
                std::vector<size_t> fused_shape = {fused_out, hidden};
                auto fused_tensor = w_qkv_param.alloc(fused_shape, acc.dtype);

                if (!fused_tensor || !fused_tensor.raw_data()) {
                    fprintf(stderr, "[C-API] ERROR: Failed to allocate w_qkv for %s\n", key.c_str());
                    continue;
                }

                // Fuse on CPU first, then transfer to GPU
                std::vector<uint8_t> fused_data(fused_tensor.byte_size());

                // Copy in order: Q, K, V
                // Each weight is transposed: HF uses [hidden, out] but TM uses [out, hidden]
                // So we need to transpose each weight during fusion
                for (size_t h = 0; h < hidden; ++h) {
                    // Copy Q row (transposed)
                    size_t q_offset = h * q_out * elem_size;
                    size_t fused_q_offset = h * elem_size;
                    std::memcpy(&fused_data[fused_q_offset], &acc.q_data[q_offset], q_out * elem_size);

                    // Copy K row (transposed)
                    size_t k_offset = h * k_out * elem_size;
                    size_t fused_k_offset = (q_out + h) * elem_size;
                    std::memcpy(&fused_data[fused_k_offset], &acc.k_data[k_offset], k_out * elem_size);

                    // Copy V row (transposed)
                    size_t v_offset = h * v_out * elem_size;
                    size_t fused_v_offset = (q_out + k_out + h) * elem_size;
                    std::memcpy(&fused_data[fused_v_offset], &acc.v_data[v_offset], v_out * elem_size);
                }

                // Copy fused data to GPU
                std::memcpy(fused_tensor.raw_data(), fused_data.data(), fused_data.size());
                auto qkv_elapsed = std::chrono::high_resolution_clock::now() - qkv_start;
                SAFETENSORS_LOG("[C-API] Fused QKV for %s: [%zu,%zu], %.0fms\n", key.c_str(), fused_out, hidden,
                               std::chrono::duration_cast<std::chrono::milliseconds>(qkv_elapsed).count());
            }

            auto qkv_total = std::chrono::high_resolution_clock::now() - load_start;
            SAFETENSORS_LOG("[C-API] QKV fusion phase complete: %.1fs total elapsed\n",
                           std::chrono::duration_cast<std::chrono::seconds>(qkv_total).count());
        }

        // Phase 2: Run all transfers in batched mode for better performance
        auto transfer_start = std::chrono::high_resolution_clock::now();
        size_t transfer_total_bytes = 0;
        for (const auto& t : transfers) transfer_total_bytes += t.size;
        SAFETENSORS_LOG("[C-API] Running batched transfers for %d tensors (%zu MB)...\n",
                       loaded_count, transfer_total_bytes / (1024 * 1024));

        auto batch_start = std::chrono::high_resolution_clock::now();
        {
            auto group = batch_copy.group();
            size_t transfer_progress = 0;
            const size_t progress_interval = std::max((size_t)100, transfers.size() / 20);
            for (const auto& transfer : transfers) {
                batch_copy(reinterpret_cast<const char*>(transfer.src), transfer.size,
                          reinterpret_cast<char*>(transfer.dst));
                transfer_progress++;
                if (transfer_progress % progress_interval == 0) {
                    auto elapsed = std::chrono::high_resolution_clock::now() - transfer_start;
                    SAFETENSORS_LOG("[C-API] Transfer progress: %zu/%zu (%.1fs elapsed)\n",
                                   transfer_progress, transfers.size(), elapsed.count() / 1e9);
                }
            }
        }
        // Group goes out of scope, then run the batch
        SAFETENSORS_LOG("[C-API] Transfer group setup done, executing Run()...\n");
        batch_copy.Run();

        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
        SAFETENSORS_LOG("[C-API] Batched transfers completed in %ldms\n", batch_ms.count());

        SAFETENSORS_LOG("[C-API] Loaded %d tensors, skipped %d from %s\n", loaded_count, skip_count, safetensors_path);

        // Trim CUDA memory pool to release unused memory back to OS
        turbomind::core::Context::device_alloc()->trim(0);
        SAFETENSORS_LOG("[C-API] Trimmed CUDA memory pool after loading %s\n", safetensors_path);
    }
    catch (const std::exception& e) {
        // Log error but continue with other files
        fprintf(stderr, "[C-API] Error loading safetensors file %s: %s\n", safetensors_path, e.what());
    }

    fprintf(stderr, "[C-API] LoadWeightsFromSafetensors EXIT: %s\n", safetensors_path);
}

// Forward declarations for debugging
namespace turbomind {
core::Param LinearWeight_debug_param(class LinearWeight* lw, const std::string& name);
}

// Map HuggingFace weight names to TurboMind module paths
// Handles MTP (Multi-Token Prediction) weight mapping for different architectures:
// - Qwen3.5 MTP: mtp.layers.X.* -> layers.X.* (shares weights with main model)
// - DeepSeek MTP: layers.{num_hidden_layers + N}.* -> MTP-specific or shared
static std::string MapHuggingFaceWeightToTurboMind(const std::string& hf_name, const HfModelConfig& hf_config)
{
    // HF format: model.language_model.layers.0.mlp.gate_proj.weight
    // TM format: layers.0.feed_forward.w1.weight
    // HF MoE format: model.layers.0.mlp.experts.0.gate_proj.weight
    // TM MoE format: layers.0.moe_ffn.experts.0.w1.weight

    std::string result = hf_name;

    // Remove "model." prefix if present
    if (result.compare(0, 6, "model.") == 0) {
        result = result.substr(6);
    }

    // Remove "language_model." prefix if present (for vision-language models)
    if (result.compare(0, 15, "language_model.") == 0) {
        result = result.substr(15);
    }

    // ========================================================
    // MTP (Multi-Token Prediction) paths
    // ========================================================
    // MTP shares weights with main model layers (mtp.layers.X.* -> layers.X.*)
    // MTP-specific params (norm, fc, pre_fc_norm_*) are NOT supported in C++ engine
    // since there's no mtp module - skip them by returning empty string
    bool is_mtp = result.compare(0, 4, "mtp.") == 0;
    if (is_mtp) {
        // Remove mtp prefix for mapping
        result = result.substr(4);  // Remove "mtp." prefix

        // MTP-specific top-level params: mtp.norm, mtp.fc, mtp.pre_fc_norm_*
        // Skip these since C++ engine has no mtp module
        if (result == "norm.weight" || result == "norm") {
            return "";  // Skip MTP-specific norm
        }
        if (result.compare(0, 3, "fc.") == 0) {
            return "";  // Skip MTP-specific fc
        }
        if (result.compare(0, 13, "pre_fc_norm_") == 0) {
            return "";  // Skip MTP-specific pre_fc_norm_*
        }

        // MTP layers share weights with main model: layers.X.*
        if (result.compare(0, 7, "layers.") == 0) {
            // Keep "layers." in the path and apply standard mappings
            // Each pattern only appears once, so no while loops needed
            replace_all(result, ".self_attn.", ".attention.");
            replace_all(result, ".mlp.experts.", ".moe_ffn.experts.");
            replace_all(result, ".mlp.", ".feed_forward.");
            replace_all(result, ".input_layernorm", ".attention_norm");
            replace_all(result, ".post_attention_layernorm", ".ffn_norm");
            replace_all(result, ".o_proj.", ".wo.");
            replace_all(result, ".gate_proj.", ".w1.");
            replace_all(result, ".up_proj.", ".w3.");
            replace_all(result, ".down_proj.", ".w2.");
            replace_all(result, ".qweight", ".weight");
            replace_all(result, ".qzeros", ".zeros");

            return result;  // Map to main model layers
        }

        return "";  // Skip unknown MTP params
    }

    // ========================================================
    // DeepSeek/GLM4 MTP paths: layers.{num_hidden_layers+N}.*
    // These are MTP predictor layers that share structure with main model layers
    // Example: DeepSeek-V2 with num_hidden_layers=61, num_nextn_predict_layers=1
    //          has weights at layers.61.self_attn.*, layers.61.mlp.* etc.
    // ========================================================
    // Check if this is a DeepSeek/GLM4 MTP layer (layers.N where N >= num_hidden_layers)
    if (hf_config.num_nextn_predict_layers > 0 && hf_config.num_hidden_layers > 0) {
        size_t layers_pos = result.find(".layers.");
        if (layers_pos != std::string::npos) {
            size_t layer_num_start = layers_pos + 8;  // ".layers." = 8 chars
            size_t layer_num_end = result.find('.', layer_num_start);

            if (layer_num_end != std::string::npos) {
                std::string layer_num_str = result.substr(layer_num_start, layer_num_end - layer_num_start);

                // Check if the layer number is >= num_hidden_layers (MTP predictor layers)
                bool is_digit = !layer_num_str.empty() &&
                                std::all_of(layer_num_str.begin(), layer_num_str.end(), ::isdigit);
                if (is_digit) {
                    int layer_num = std::stoi(layer_num_str);
                    if (layer_num >= hf_config.num_hidden_layers) {
                        // This is an MTP predictor layer
                        // MTP-specific params: embed_tokens, enorm, hnorm, shared_head, eh_proj, rotary_emb
                        // These are NOT supported in C++ engine - skip them
                        if (result == "embed_tokens.weight") {
                            return "";  // Skip MTP-specific embeddings
                        }
                        if (result == "enorm.weight") {
                            return "";  // Skip MTP embedding norm
                        }
                        if (result == "hnorm.weight") {
                            return "";  // Skip MTP hidden norm
                        }
                        if (result == "eh_proj.weight") {
                            return "";  // Skip MTP embedding-hidden projection
                        }
                        if (result.find("shared_head.") == 0) {
                            return "";  // Skip MTP shared head components
                        }
                        if (result.find("rotary_emb.") == 0) {
                            return "";  // Skip MTP rotary embeddings
                        }

                        // For transformer layer components (attention, FFN, layer norms)
                        // DeepSeek MTP shares structure with main model layers
                        // Map MTP layer to main model layer 0 (or cycle through main model layers)
                        // layers.{num_hidden_layers+N}.self_attn.* -> layers.{N % num_hidden_layers}.attention.*
                        int target_layer = layer_num % hf_config.num_hidden_layers;
                        std::string target_layer_str = std::to_string(target_layer);

                        // Replace the layer number in the path
                        // layers.N.xxx -> layers.{target_layer}.xxx
                        result = result.substr(0, layer_num_start) + target_layer_str + result.substr(layer_num_end);

                        // Apply standard MTP mappings and continue with normal processing
                        replace_all(result, ".self_attn.", ".attention.");
                        replace_all(result, ".mlp.experts.", ".moe_ffn.experts.");
                        replace_all(result, ".mlp.", ".feed_forward.");
                        replace_all(result, ".input_layernorm", ".attention_norm");
                        replace_all(result, ".post_attention_layernorm", ".ffn_norm");
                        replace_all(result, ".o_proj.", ".wo.");
                        replace_all(result, ".gate_proj.", ".w1.");
                        replace_all(result, ".up_proj.", ".w3.");
                        replace_all(result, ".down_proj.", ".w2.");
                        replace_all(result, ".qweight", ".weight");
                        replace_all(result, ".qzeros", ".zeros");

                        return result;  // Return mapped path
                    }
                }
            }
        }
    }

    // ========================================================
    // Top-level params (no dots before them)
    // ========================================================
    // lm_head.weight -> output.weight
    if (result.compare(0, 8, "lm_head.") == 0) {
        result = "output." + result.substr(8);  // "lm_head." = 8 chars
    }
    // embed_tokens.weight -> tok_embeddings (direct param on model_weight)
    if (result.compare(0, 12, "embed_tokens.") == 0) {
        result = "tok_embeddings";  // Return just the param name, not "tok_embeddings.weight"
    }

    // ========================================================
    // MoE-specific mappings (must be before general mlp -> feed_forward)
    // ========================================================
    // Handle MoE expert paths: .mlp.experts.N.<proj> -> .moe_ffn.experts.N.<proj>
    // This pattern matches: layers.X.mlp.experts.Y.gate_proj.weight
    replace_all(result, ".mlp.experts.", ".moe_ffn.experts.");

    // Handle MoE gate: .mlp.gate.weight -> .moe_ffn.gate.weight
    // (This is the router gate, not to be confused with gate_proj)
    // Pattern: layers.X.mlp.gate.weight -> layers.X.moe_ffn.gate.weight
    // We need to be careful not to match .gate_proj.
    size_t mlp_gate_pos = result.find(".mlp.gate.");
    if (mlp_gate_pos != std::string::npos) {
        // Check if this is actually .mlp.gate_proj. (skip if so)
        size_t gate_proj_pos = result.find(".gate_proj.", mlp_gate_pos);
        size_t next_dot = result.find('.', mlp_gate_pos + 10);
        size_t weight_pos = result.find("weight", mlp_gate_pos);
        // Only replace if we find "weight" before the next dot and no gate_proj
        if (gate_proj_pos == std::string::npos &&
            weight_pos != std::string::npos &&
            (next_dot == std::string::npos || weight_pos < next_dot)) {
            result.replace(mlp_gate_pos, 10, ".moe_ffn.gate.");
        }
    }

    // ========================================================
    // DeltaNet (linear_attn) specific mappings
    // ========================================================
    // Map HF DeltaNet layer paths to TurboMind linear_attn paths
    // HF format: layers.X.self_attn.in_proj.qkv.weight
    // TM format: layers.X.linear_attn.in_proj_qkv.weight
    // Apply these BEFORE the general self_attn -> attention replacement
    replace_all(result, ".self_attn.in_proj.qkv.", ".linear_attn.in_proj_qkv.");
    replace_all(result, ".self_attn.in_proj.z.", ".linear_attn.in_proj_z.");
    replace_all(result, ".self_attn.in_proj.a.", ".linear_attn.in_proj_a.");
    replace_all(result, ".self_attn.in_proj.b.", ".linear_attn.in_proj_b.");
    replace_all(result, ".self_attn.in_proj.weight", ".linear_attn.in_proj_all.weight");
    replace_all(result, ".linear_attn.linear_out_proj.", ".linear_attn.out_proj.");
    replace_all(result, ".self_attn.linear_out_proj.", ".linear_attn.out_proj.");
    replace_all(result, ".linear_attn.conv1d.weight", ".linear_attn.conv1d");

    // self_attn -> attention (for standard full attention layers)
    replace_all(result, ".self_attn.", ".attention.");

    // mlp -> feed_forward (only for non-MoE models)
    // Skip if we already have moe_ffn
    size_t mlp_pos = result.find(".mlp.");
    while (mlp_pos != std::string::npos) {
        // Don't replace if it's part of moe_ffn
        if (mlp_pos >= 5 && result.compare(mlp_pos - 5, 8, ".moe_ffn.") != 0) {
            result.replace(mlp_pos, 5, ".feed_forward.");
            mlp_pos = result.find(".mlp.", mlp_pos + 14);  // Skip the replacement
        } else {
            break;
        }
    }

    // Layer norm replacements
    replace_all(result, ".input_layernorm", ".attention_norm");
    replace_all(result, ".post_attention_layernorm", ".ffn_norm");

    // Map HF q_proj/k_proj/v_proj to TM fused w_qkv
    // HF stores separate Q/K/V weights, but TM uses a fused w_qkv weight
    // Each of q_proj, k_proj, v_proj maps to w_qkv for weight loading
    replace_all(result, ".q_proj.", ".w_qkv.");
    replace_all(result, ".k_proj.", ".w_qkv.");
    replace_all(result, ".v_proj.", ".w_qkv.");
    replace_all(result, ".o_proj.", ".wo.");

    // FFN projections (applies to both standard FFN and MoE experts)
    replace_all(result, ".gate_proj.", ".w1.");
    replace_all(result, ".up_proj.", ".w3.");
    replace_all(result, ".down_proj.", ".w2.");

    // Note: DeltaNet (linear_attn) specific mappings are handled earlier
    // in the function (before self_attn -> attention replacement)

    // AWQ quantization parameters (qweight -> weight, etc.)
    // These must be last so .weight suffix is already established
    replace_all(result, ".qweight", ".weight");
    replace_all(result, ".qzeros", ".zeros");
    replace_all(result, ".weight_scale", ".scales");
    replace_all(result, ".weight_zero", ".zeros");

    return result;
}

int TM_TurboMind_InitFromHF(
    TM_TurboMind* tm,
    int device_id,
    const char* model_dir,
    int trust_remote_code,
    int session_len)
{
    // Use Python bridge for HF model loading.
    // The Python TurboMind API already supports HF safetensors loading.
    // We'll call it via a simple Python script.
    std::string python_cmd = "python3 -c \""
        "import sys; "
        "sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/lmdeploy'); "
        "from lmdeploy.turbomind.turbomind import TurboMind; "
        "from lmdeploy.messages import TurbomindEngineConfig; "
        "cfg = TurbomindEngineConfig(session_len=" + std::to_string(session_len) + ", max_batch_size=8); "
        "tm = TurboMind('" + std::string(model_dir) + "', engine_config=cfg, trust_remote_code=" + (trust_remote_code ? "True" : "False") + "); "
        "print('SUCCESS')"
        "\" 2>&1";

    FILE* pipe = popen(python_cmd.c_str(), "r");
    if (!pipe) {
        SetError(TM_ERR_RUNTIME, "Failed to run Python model loading script");
        return TM_ERR_RUNTIME;
    }

    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    int status = pclose(pipe);

    // Check if Python succeeded
    if (result.find("SUCCESS") == std::string::npos) {
        std::string error_msg = "Python model loading failed. Output:\n" + result;
        SetError(TM_ERR_RUNTIME, error_msg.c_str());
        return TM_ERR_RUNTIME;
    }

    // Python succeeded - now initialize the C++ TurboMind instance
    // The Python script would have populated the model structure
    // For now, we need to return an error that the user should use Python bridge
    SetError(TM_ERR_NOT_IMPLEMENTED,
        "HF model loading requires Python bridge. "
        "Please use the Python API (lmdeploy.turbomind.TurboMind) or PyTorch backend. "
        "The C API currently only supports TurboMind-converted models.");
    return TM_ERR_NOT_IMPLEMENTED;
}

int TM_TurboMind_InitFromPath(TM_TurboMind* tm, int device_id, const char* model_dir, int trust_remote_code)
{
    // Disable stderr buffering for real-time logging
    setbuf(stderr, NULL);

    debug_log("[InitFromPath] ENTER\n");
    if (!tm || !model_dir) {
        debug_log("[InitFromPath] Invalid args\n");
        SetError(TM_ERR_INVALID_ARG, "tm and model_dir must not be NULL");
        return TM_ERR_INVALID_ARG;
    }

    try {
        const int index = 0;  // For single-GPU, always use index 0

        // Step 1: Create CUDA context
        tm->instance->CreateContext(index);

        // Step 2: Create ModelRoot sentinel
        auto* root = tm->instance->CreateRoot(index);
        if (!root) {
            SetError(TM_ERR_RUNTIME, "CreateRoot failed");
            return TM_ERR_RUNTIME;
        }

        debug_log("[InitFromPath] Step 2: CreateRoot OK\n");

        // Step 3: Parse HuggingFace config.json
        HfModelConfig hf_config = ParseHfConfig(std::string(model_dir));

        char cfg_log[128];
        snprintf(cfg_log, sizeof(cfg_log), "[InitFromPath] hs=%d nl=%d na=%d v=%d\n",
                 hf_config.hidden_size, hf_config.num_hidden_layers,
                 hf_config.num_attention_heads, hf_config.vocab_size);
        debug_log(cfg_log);

        // Create ModelWeightConfig with appropriate settings
        turbomind::core::ModelWeightConfig weight_cfg;
        weight_cfg.hidden_units = hf_config.hidden_size;
        weight_cfg.tp_size = 1;
        weight_cfg.tp_rank = 0;

        // Determine data type based on quantization
        if (hf_config.is_awq) {
            weight_cfg.data_type = turbomind::DataType::kFloat16;
        } else {
            weight_cfg.data_type = turbomind::DataType::kFloat16;
        }

        // Create ModelWeight via Module::create
        auto weight_module = turbomind::core::Module::create(weight_cfg);
        if (!weight_module) {
            SetError(TM_ERR_RUNTIME, "Failed to create ModelWeight module");
            return TM_ERR_RUNTIME;
        }

        fprintf(stderr, "[C-API] Parsed config: hidden_size=%d, num_layers=%d, num_heads=%d, num_kv_heads=%d, is_awq=%d\n",
                hf_config.hidden_size, hf_config.num_hidden_layers, hf_config.num_attention_heads,
                hf_config.num_key_value_heads, hf_config.is_awq);
        fflush(stderr);

        // Attach ModelWeight to ModelRoot as 'text_model' child FIRST
        // This moves weight_module into the ModelRoot, so weight_module is no longer valid
        auto* model_root = static_cast<turbomind::ModelRoot*>(root);
        auto* result = model_root->add_child("text_model", std::move(weight_module));
        if (!result) {
            SetError(TM_ERR_RUNTIME, "Failed to attach ModelWeight to ModelRoot");
            return TM_ERR_RUNTIME;
        }

        // Get ModelWeight pointer from the ModelRoot (now owned by ModelRoot)
        auto* model_weight = model_root->text_model_ptr();
        if (!model_weight) {
            SetError(TM_ERR_RUNTIME, "Failed to get ModelWeight from ModelRoot");
            return TM_ERR_RUNTIME;
        }

        // All GPU allocations must happen under ContextGuard.
        // Create the guard BEFORE any GPU memory allocation.
        // The guard's destructor pushes CUDA context + allocator, and pops on scope exit.
        auto ctx_guard = model_root->context();

        // DEBUG: Verify ContextGuard is working correctly
        {
            char dbg[512];
            // Check if context() returns valid stream and allocator
            auto& alloc = turbomind::core::Context::device_alloc();
            snprintf(dbg, sizeof(dbg),
                "[InitFromPath] DEBUG: ctx_guard created, Context::device_alloc().valid=%d, device.type=%d\n",
                (bool)alloc, (int)(alloc->device().type));
            debug_log(dbg);
            fprintf(stderr, "%s", dbg);
            fflush(stderr);
        }

        // 1. Create and add tok_embeddings param
        // Shape: [vocab_size, hidden_size]
        std::vector<size_t> tok_emb_shape = {(size_t)hf_config.vocab_size, (size_t)hf_config.hidden_size};
        auto tok_emb_param = model_weight->param("tok_embeddings");
        if (tok_emb_param) {
            tok_emb_param.alloc(tok_emb_shape, weight_cfg.data_type);
        }

        // 2. Create and add norm child (NormWeight)
        turbomind::core::NormConfig norm_cfg;
        norm_cfg.dim = hf_config.hidden_size;
        norm_cfg.data_type = weight_cfg.data_type;
        norm_cfg.norm_eps = 1e-6f;  // Default RMS norm eps
        auto norm_module = turbomind::core::Module::create(norm_cfg);
        if (norm_module) {
            model_weight->add_child("norm", std::move(norm_module));
        }

        // 3. Create and add output child (LinearWeight)
        auto output_cfg = CreateAwqLinearConfig(
            hf_config.hidden_size, hf_config.vocab_size,
            weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
        auto output_module = turbomind::core::Module::create(output_cfg);
        if (output_module) {
            model_weight->add_child("output", std::move(output_module));
        }

        // 4. Create and add layers ModuleList
        turbomind::core::ModuleListConfig layers_cfg;
        auto layers_list_unique = turbomind::core::Module::create(layers_cfg);
        auto* layers_list = static_cast<turbomind::core::ModuleList*>(layers_list_unique.get());

        // Calculate head dimensions
        int head_dim = hf_config.hidden_size / hf_config.num_attention_heads;

        fprintf(stderr, "[C-API] Creating %d decoder layers (hidden_size=%d, num_heads=%d, head_dim=%d)\n",
                hf_config.num_hidden_layers, hf_config.hidden_size, hf_config.num_attention_heads, head_dim);
        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf), "[C-API] Creating %d layers (hs=%d, nh=%d, hd=%d)\n",
                 hf_config.num_hidden_layers, hf_config.hidden_size, hf_config.num_attention_heads, head_dim);
        debug_log(log_buf);

        // Create each decoder layer
        for (int layer_idx = 0; layer_idx < hf_config.num_hidden_layers; ++layer_idx) {
            // Create DecoderLayerWeight
            turbomind::core::DecoderLayerConfig layer_cfg;
            auto layer_module = turbomind::core::Module::create(layer_cfg);
            if (!layer_module) {
                continue;  // Skip if creation failed
            }

            auto* decoder_layer = static_cast<turbomind::DecoderLayerWeight*>(layer_module.get());

            // 4a. Create attention_norm (NormWeight)
            turbomind::core::NormConfig attn_norm_cfg;
            attn_norm_cfg.dim = hf_config.hidden_size;
            attn_norm_cfg.data_type = weight_cfg.data_type;
            attn_norm_cfg.norm_eps = 1e-6f;
            auto attn_norm_module = turbomind::core::Module::create(attn_norm_cfg);
            if (attn_norm_module) {
                decoder_layer->add_child("attention_norm", std::move(attn_norm_module));
            }

            // 4b. Create ffn_norm (NormWeight)
            turbomind::core::NormConfig ffn_norm_cfg;
            ffn_norm_cfg.dim = hf_config.hidden_size;
            ffn_norm_cfg.data_type = weight_cfg.data_type;
            ffn_norm_cfg.norm_eps = 1e-6f;
            auto ffn_norm_module = turbomind::core::Module::create(ffn_norm_cfg);
            if (ffn_norm_module) {
                decoder_layer->add_child("ffn_norm", std::move(ffn_norm_module));
            }

            // 4c. Create attention (AttentionWeight) with its child modules
            turbomind::core::AttentionConfig attn_cfg;
            attn_cfg.hidden_dim = hf_config.hidden_size;
            attn_cfg.head_dim = head_dim;
            attn_cfg.head_num = hf_config.num_attention_heads;
            attn_cfg.kv_head_num = hf_config.num_key_value_heads;
            attn_cfg.kv_lora_rank = 0;
            attn_cfg.q_lora_rank = 0;
            attn_cfg.qk_rope_dim = 0;
            attn_cfg.v_head_dim = 0;
            attn_cfg.tp_size = 1;
            attn_cfg.tp_rank = 0;
            attn_cfg.data_type = weight_cfg.data_type;
            attn_cfg.window_size = 0;
            attn_cfg.output_gate = false;  // Set to true for Qwen3.5
            attn_cfg.softmax_scale = 0.0f;
            attn_cfg.use_logn_attn = false;
            attn_cfg.rope = turbomind::core::RopeConfig{};
            auto attn_module = turbomind::core::Module::create(attn_cfg);
            if (attn_module) {
                auto* attn = static_cast<turbomind::AttentionWeight*>(attn_module.get());

                // Create fused w_qkv LinearWeight child (matches TurboMind's AttentionWeight)
                // HF uses separate q_proj/k_proj/v_proj, but TM uses fused w_qkv
                // The fused output dimension = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
                const int qkv_out_dim = hf_config.num_attention_heads * head_dim
                                      + 2 * hf_config.num_key_value_heads * head_dim;
                auto w_qkv_cfg = CreateAwqLinearConfig(
                    hf_config.hidden_size, qkv_out_dim,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto w_qkv_module = turbomind::core::Module::create(w_qkv_cfg);
                if (w_qkv_module) {
                    attn->add_child("w_qkv", std::move(w_qkv_module));
                }

                // Create wo LinearWeight child
                auto wo_cfg = CreateAwqLinearConfig(
                    hf_config.num_attention_heads * head_dim, hf_config.hidden_size,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto wo_module = turbomind::core::Module::create(wo_cfg);
                if (wo_module) {
                    attn->add_child("wo", std::move(wo_module));
                }

                decoder_layer->add_child("attention", std::move(attn_module));
            }

            // 4d. Create feed_forward (FfnWeight) with its child modules
            turbomind::core::FfnConfig ffn_cfg;
            ffn_cfg.hidden_dim = hf_config.hidden_size;
            ffn_cfg.inter_size = hf_config.intermediate_size;
            ffn_cfg.act_type = 0;  // SiLU
            ffn_cfg.fuse_silu = true;
            ffn_cfg.is_expert = hf_config.num_local_experts > 0;
            ffn_cfg.data_type = weight_cfg.data_type;
            ffn_cfg.tp_size = 1;
            ffn_cfg.tp_rank = 0;
            auto ffn_module = turbomind::core::Module::create(ffn_cfg);
            if (ffn_module) {
                auto* ffn = static_cast<turbomind::FfnWeight*>(ffn_module.get());

                // Create w1 LinearWeight child (gate_proj)
                auto w1_cfg = CreateAwqLinearConfig(
                    hf_config.hidden_size, hf_config.intermediate_size,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto w1_module = turbomind::core::Module::create(w1_cfg);
                if (w1_module) {
                    ffn->add_child("w1", std::move(w1_module));
                }

                // Create w3 LinearWeight child (up_proj)
                auto w3_cfg = CreateAwqLinearConfig(
                    hf_config.hidden_size, hf_config.intermediate_size,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto w3_module = turbomind::core::Module::create(w3_cfg);
                if (w3_module) {
                    ffn->add_child("w3", std::move(w3_module));
                }

                // Create w2 LinearWeight child (down_proj)
                auto w2_cfg = CreateAwqLinearConfig(
                    hf_config.intermediate_size, hf_config.hidden_size,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto w2_module = turbomind::core::Module::create(w2_cfg);
                if (w2_module) {
                    ffn->add_child("w2", std::move(w2_module));
                }

                decoder_layer->add_child("feed_forward", std::move(ffn_module));
            }

            // 4e. Create moe_ffn (MoeWeight) with its child modules if this is a MoE model
            if (hf_config.num_local_experts > 0) {
                turbomind::core::MoeConfig moe_cfg;
                moe_cfg.expert_num = hf_config.num_local_experts;
                moe_cfg.experts_per_token = hf_config.num_experts_per_tok;
                moe_cfg.act_type = 0;  // SiLU
                moe_cfg.fuse_silu = true;
                moe_cfg.norm_topk_prob = true;
                moe_cfg.topk_method = "greedy";
                moe_cfg.scoring_func = "softmax";
                moe_cfg.topk_group = 0;
                moe_cfg.n_group = 0;
                moe_cfg.router_n_groups = 1;
                moe_cfg.routed_scale = 1.0;
                moe_cfg.data_type = weight_cfg.data_type;
                auto moe_module = turbomind::core::Module::create(moe_cfg);
                if (moe_module) {
                    auto* moe = static_cast<turbomind::MoeWeight*>(moe_module.get());

                    // Create gate LinearWeight child
                    auto gate_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, hf_config.num_local_experts,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto gate_module = turbomind::core::Module::create(gate_cfg);
                    if (gate_module) {
                        moe->add_child("gate", std::move(gate_module));
                    }

                    // Create experts ModuleList child
                    turbomind::core::ModuleListConfig experts_cfg;
                    auto experts_list_unique = turbomind::core::Module::create(experts_cfg);
                    auto* experts_list = static_cast<turbomind::core::ModuleList*>(experts_list_unique.get());

                    // Create each expert FfnWeight
                    for (int expert_idx = 0; expert_idx < hf_config.num_local_experts; ++expert_idx) {
                        turbomind::core::FfnConfig expert_ffn_cfg;
                        expert_ffn_cfg.hidden_dim = hf_config.hidden_size;
                        expert_ffn_cfg.inter_size = hf_config.intermediate_size;
                        expert_ffn_cfg.act_type = 0;  // SiLU
                        expert_ffn_cfg.fuse_silu = true;
                        expert_ffn_cfg.is_expert = true;
                        expert_ffn_cfg.data_type = weight_cfg.data_type;
                        expert_ffn_cfg.tp_size = 1;
                        expert_ffn_cfg.tp_rank = 0;
                        auto expert_ffn_module = turbomind::core::Module::create(expert_ffn_cfg);
                        if (expert_ffn_module) {
                            auto* expert_ffn = static_cast<turbomind::FfnWeight*>(expert_ffn_module.get());

                            // Create w1 LinearWeight child for expert
                            auto expert_w1_cfg = CreateAwqLinearConfig(
                                hf_config.hidden_size, hf_config.intermediate_size,
                                weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                            auto expert_w1_module = turbomind::core::Module::create(expert_w1_cfg);
                            if (expert_w1_module) {
                                expert_ffn->add_child("w1", std::move(expert_w1_module));
                            }

                            // Create w3 LinearWeight child for expert
                            auto expert_w3_cfg = CreateAwqLinearConfig(
                                hf_config.hidden_size, hf_config.intermediate_size,
                                weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                            auto expert_w3_module = turbomind::core::Module::create(expert_w3_cfg);
                            if (expert_w3_module) {
                                expert_ffn->add_child("w3", std::move(expert_w3_module));
                            }

                            // Create w2 LinearWeight child for expert
                            auto expert_w2_cfg = CreateAwqLinearConfig(
                                hf_config.intermediate_size, hf_config.hidden_size,
                                weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                            auto expert_w2_module = turbomind::core::Module::create(expert_w2_cfg);
                            if (expert_w2_module) {
                                expert_ffn->add_child("w2", std::move(expert_w2_module));
                            }

                            // Add expert to experts ModuleList
                            experts_list->add_child(std::to_string(expert_idx), std::move(expert_ffn_module));
                        }
                    }

                    // Add experts to MoeWeight
                    moe->add_child("experts", std::move(experts_list_unique));

                    decoder_layer->add_child("moe_ffn", std::move(moe_module));
                }
            }

            // 4f. Create linear_attn (DeltaNetWeight) if this layer uses linear attention
            // or create attention (AttentionWeight) if this layer uses full attention
            bool is_linear_attn = layer_idx < (int)hf_config.layer_is_linear_attn.size() &&
                                  hf_config.layer_is_linear_attn[layer_idx];

            if (is_linear_attn) {
                turbomind::core::DeltaNetConfig delta_cfg;
                delta_cfg.hidden_dim = hf_config.hidden_size;
                delta_cfg.num_k_heads = hf_config.num_attention_heads;
                delta_cfg.num_v_heads = hf_config.num_key_value_heads;
                delta_cfg.key_head_dim = head_dim;
                delta_cfg.value_head_dim = head_dim;
                delta_cfg.d_conv = 4;
                delta_cfg.data_type = weight_cfg.data_type;
                delta_cfg.tp_size = 1;
                delta_cfg.tp_rank = 0;
                auto delta_module = turbomind::core::Module::create(delta_cfg);
                if (delta_module) {
                    auto* delta = static_cast<turbomind::DeltaNetWeight*>(delta_module.get());

                    // Calculate the correct fused input projection dimension
                    // Layout: [qkv | z | b | a]
                    // qkv: 2*num_k_heads*key_head_dim + num_v_heads*value_head_dim
                    // z: num_v_heads*value_head_dim
                    // b: num_v_heads
                    // a: num_v_heads
                    const int num_k_heads = hf_config.num_attention_heads;
                    const int num_v_heads = hf_config.num_key_value_heads;
                    const int key_head_dim = head_dim;
                    const int value_head_dim = head_dim;
                    const int qkv_out = 2 * num_k_heads * key_head_dim + num_v_heads * value_head_dim;
                    const int z_out = num_v_heads * value_head_dim;
                    const int a_out = num_v_heads;
                    const int b_out = num_v_heads;
                    const int fused_out = qkv_out + z_out + a_out + b_out;

                    // Create separate projection LinearWeight children (for HF weight loading)
                    // These will be fused into in_proj_all during prepare()
                    char log_buf_dn[512];

                    // Linear attention layers use BF16 weights (not AWQ quantized).
                    // AWQ config has "modules_to_not_convert" including "linear_attn",
                    // so these layers must use plain format, not AWQ kUint4 format.
                    auto in_proj_qkv_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, qkv_out,
                        weight_cfg.data_type, false, hf_config.awq_group_size);
                    auto in_proj_qkv_module = turbomind::core::Module::create(in_proj_qkv_cfg);
                    snprintf(log_buf_dn, sizeof(log_buf_dn),
                        "[DeltaNet] layer=%d Module::create(in_proj_qkv) -> %p\n",
                        layer_idx, static_cast<void*>(in_proj_qkv_module.get()));
                    debug_log(log_buf_dn);
                    if (in_proj_qkv_module) {
                        auto* add_result = delta->add_child("in_proj_qkv", std::move(in_proj_qkv_module));
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d add_child(in_proj_qkv) -> %p\n",
                            layer_idx, static_cast<void*>(add_result));
                        debug_log(log_buf_dn);
                        auto* child_check = delta->child("in_proj_qkv");
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d child(in_proj_qkv) -> %p\n",
                            layer_idx, static_cast<void*>(child_check));
                        debug_log(log_buf_dn);
                    }

                    auto in_proj_z_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, z_out,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto in_proj_z_module = turbomind::core::Module::create(in_proj_z_cfg);
                    snprintf(log_buf_dn, sizeof(log_buf_dn),
                        "[DeltaNet] layer=%d Module::create(in_proj_z) -> %p\n",
                        layer_idx, static_cast<void*>(in_proj_z_module.get()));
                    debug_log(log_buf_dn);
                    if (in_proj_z_module) {
                        auto* add_result = delta->add_child("in_proj_z", std::move(in_proj_z_module));
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d add_child(in_proj_z) -> %p\n",
                            layer_idx, static_cast<void*>(add_result));
                        debug_log(log_buf_dn);
                        auto* child_check = delta->child("in_proj_z");
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d child(in_proj_z) -> %p\n",
                            layer_idx, static_cast<void*>(child_check));
                        debug_log(log_buf_dn);
                    }

                    auto in_proj_a_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, a_out,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto in_proj_a_module = turbomind::core::Module::create(in_proj_a_cfg);
                    snprintf(log_buf_dn, sizeof(log_buf_dn),
                        "[DeltaNet] layer=%d Module::create(in_proj_a) -> %p\n",
                        layer_idx, static_cast<void*>(in_proj_a_module.get()));
                    debug_log(log_buf_dn);
                    if (in_proj_a_module) {
                        auto* add_result = delta->add_child("in_proj_a", std::move(in_proj_a_module));
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d add_child(in_proj_a) -> %p\n",
                            layer_idx, static_cast<void*>(add_result));
                        debug_log(log_buf_dn);
                        auto* child_check = delta->child("in_proj_a");
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d child(in_proj_a) -> %p\n",
                            layer_idx, static_cast<void*>(child_check));
                        debug_log(log_buf_dn);
                    }

                    auto in_proj_b_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, b_out,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto in_proj_b_module = turbomind::core::Module::create(in_proj_b_cfg);
                    snprintf(log_buf_dn, sizeof(log_buf_dn),
                        "[DeltaNet] layer=%d Module::create(in_proj_b) -> %p\n",
                        layer_idx, static_cast<void*>(in_proj_b_module.get()));
                    debug_log(log_buf_dn);
                    if (in_proj_b_module) {
                        auto* add_result = delta->add_child("in_proj_b", std::move(in_proj_b_module));
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d add_child(in_proj_b) -> %p\n",
                            layer_idx, static_cast<void*>(add_result));
                        debug_log(log_buf_dn);
                        auto* child_check = delta->child("in_proj_b");
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d child(in_proj_b) -> %p\n",
                            layer_idx, static_cast<void*>(child_check));
                        debug_log(log_buf_dn);
                    }

                    // Create in_proj_all LinearWeight child (fused, used during forward)
                    auto in_proj_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, fused_out,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto in_proj_module = turbomind::core::Module::create(in_proj_cfg);
                    snprintf(log_buf_dn, sizeof(log_buf_dn),
                        "[DeltaNet] layer=%d Module::create(in_proj_all) -> %p\n",
                        layer_idx, static_cast<void*>(in_proj_module.get()));
                    debug_log(log_buf_dn);
                    if (in_proj_module) {
                        auto* add_result = delta->add_child("in_proj_all", std::move(in_proj_module));
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d add_child(in_proj_all) -> %p\n",
                            layer_idx, static_cast<void*>(add_result));
                        debug_log(log_buf_dn);
                        auto* child_check = delta->child("in_proj_all");
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d child(in_proj_all) -> %p\n",
                            layer_idx, static_cast<void*>(child_check));
                        debug_log(log_buf_dn);
                    }

                    // Create out_proj LinearWeight child
                    auto out_proj_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, hf_config.hidden_size,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto out_proj_module = turbomind::core::Module::create(out_proj_cfg);
                    snprintf(log_buf_dn, sizeof(log_buf_dn),
                        "[DeltaNet] layer=%d Module::create(out_proj) -> %p\n",
                        layer_idx, static_cast<void*>(out_proj_module.get()));
                    debug_log(log_buf_dn);
                    if (out_proj_module) {
                        auto* add_result = delta->add_child("out_proj", std::move(out_proj_module));
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d add_child(out_proj) -> %p\n",
                            layer_idx, static_cast<void*>(add_result));
                        debug_log(log_buf_dn);
                        auto* child_check = delta->child("out_proj");
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d child(out_proj) -> %p\n",
                            layer_idx, static_cast<void*>(child_check));
                        debug_log(log_buf_dn);
                    }

                    // Create norm NormWeight child
                    turbomind::core::NormConfig delta_norm_cfg;
                    delta_norm_cfg.dim = hf_config.hidden_size;
                    delta_norm_cfg.data_type = weight_cfg.data_type;
                    delta_norm_cfg.norm_eps = 1e-6f;
                    auto delta_norm_module = turbomind::core::Module::create(delta_norm_cfg);
                    snprintf(log_buf_dn, sizeof(log_buf_dn),
                        "[DeltaNet] layer=%d Module::create(norm) -> %p\n",
                        layer_idx, static_cast<void*>(delta_norm_module.get()));
                    debug_log(log_buf_dn);
                    if (delta_norm_module) {
                        auto* add_result = delta->add_child("norm", std::move(delta_norm_module));
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d add_child(norm) -> %p\n",
                            layer_idx, static_cast<void*>(add_result));
                        debug_log(log_buf_dn);
                        auto* child_check = delta->child("norm");
                        snprintf(log_buf_dn, sizeof(log_buf_dn),
                            "[DeltaNet] layer=%d child(norm) -> %p\n",
                            layer_idx, static_cast<void*>(child_check));
                        debug_log(log_buf_dn);
                    }

                    debug_log("[DeltaNet] All children created, adding linear_attn to decoder_layer\n");

                    decoder_layer->add_child("linear_attn", std::move(delta_module));
                }
            }

            // Add layer to ModuleList
            layers_list->add_child(std::to_string(layer_idx), std::move(layer_module));

            char log_buf2[128];
            snprintf(log_buf2, sizeof(log_buf2), "[C-API] Added layer %d to ModuleList\n", layer_idx);
            debug_log(log_buf2);
        }

        char log_buf3[128];
        snprintf(log_buf3, sizeof(log_buf3), "[C-API] ModuleList size after loop: %d\n", layers_list->size());
        debug_log(log_buf3);

        // Add layers to ModelWeight
        auto* layers_result = model_weight->add_child("layers", std::move(layers_list_unique));
        if (!layers_result) {
            SetError(TM_ERR_RUNTIME, "Failed to add layers child to ModelWeight");
            return TM_ERR_RUNTIME;
        }

        fprintf(stderr, "[C-API] After add_child('layers'), model_weight->layers size = %d\n",
                model_weight->layers ? model_weight->layers->size() : -1);
        fflush(stderr);

        char log_buf4[128];
        snprintf(log_buf4, sizeof(log_buf4), "[C-API] After add_child: layers ptr=%p, size=%d\n",
                 (void*)model_weight->layers.get(), model_weight->layers ? model_weight->layers->size() : -1);
        debug_log(log_buf4);

        // Verify layers were added
        if (!model_weight->layers) {
            SetError(TM_ERR_RUNTIME, "layers child is null after add_child");
            return TM_ERR_RUNTIME;
        }

        // NOTE: ModelWeight is already attached to ModelRoot via add_child_raw above
        // No need to attach again here - weight_module is already moved

        fprintf(stderr, "[C-API] Before FindSafetensorsFiles...\n");
        fflush(stderr);

        // Step 4: Load weights from safetensors files
        // Find all safetensors files in the model directory
        std::vector<std::string> safetensors_files = FindSafetensorsFiles(model_dir);

        fprintf(stderr, "[C-API] Found %zu safetensors files\n", safetensors_files.size());
        for (const auto& f : safetensors_files) {
            fprintf(stderr, "[C-API]  - %s\n", f.c_str());
        }
        fflush(stderr);

        if (safetensors_files.empty()) {
            SetError(TM_ERR_RUNTIME, "No safetensors files found in model directory");
            return TM_ERR_RUNTIME;
        }

        // Load weights from each safetensors file
        for (const auto& st_file : safetensors_files) {
            fprintf(stderr, "[C-API] Loading file: %s\n", st_file.c_str());
            fflush(stderr);

            // DEBUG: Verify ctx_guard is still active before calling LoadWeightsFromSafetensors
            {
                auto& alloc = turbomind::core::Context::device_alloc();
                fprintf(stderr, "[DEBUG] Before LoadWeights: device_alloc valid=%d type=%d\n",
                        (bool)alloc, (int)(alloc->device().type));
                fflush(stderr);
            }

            LoadWeightsFromSafetensors(model_weight, st_file.c_str(), hf_config);
            fprintf(stderr, "[C-API] Finished loading file: %s\n", st_file.c_str());
            fflush(stderr);
        }

        // Step 5: Process weights (moves weights to GPU and calls prepare)
        tm->instance->ProcessWeights(index);

        // Step 6: Create inference engine
        tm->instance->CreateEngine(index);

        return TM_OK;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return TM_ERR_RUNTIME;
    }
}

int TM_TurboMind_GetScheduleMetrics(
    TM_TurboMind* tm,
    int index,
    int* total_seqs,
    int* active_seqs,
    int* waiting_seqs,
    int* total_blocks,
    int* active_blocks,
    int* cached_blocks,
    int* free_blocks)
{
    auto metrics = tm->instance->GetScheduleMetrics(index);
    if (!metrics) {
        return -1;
    }
    if (total_seqs)   *total_seqs   = metrics->total_seqs;
    if (active_seqs)  *active_seqs  = metrics->active_seqs;
    if (waiting_seqs) *waiting_seqs = metrics->waiting_seqs;
    if (total_blocks)  *total_blocks  = metrics->total_blocks;
    if (active_blocks) *active_blocks = metrics->active_blocks;
    if (cached_blocks) *cached_blocks = metrics->cached_blocks;
    if (free_blocks)   *free_blocks   = metrics->free_blocks;
    return 0;
}

// ============================================================
// Tensor / TensorMap
// ============================================================

namespace {

template<typename T>
void SetTensorCommon(turbomind::core::TensorMap* map, const char* name, const T* data, int ndim, const int64_t* shape,
                     turbomind::DeviceType device)
{
    std::vector<turbomind::core::ssize_t> shape_vec(shape, shape + ndim);
    turbomind::Layout layout{shape_vec};
    turbomind::DataType dtype = turbomind::data_type_v<T>;

    auto tensor = turbomind::core::Tensor(
        const_cast<T*>(data),
        layout,
        dtype,
        device);

    (*map)[name] = std::move(tensor);
}

}  // anonymous namespace

struct TM_TensorMap {
    turbomind::core::TensorMap map;

    TM_TensorMap* set(const char* name, turbomind::core::Tensor tensor)
    {
        map[name] = std::move(tensor);
        return this;
    }
};

TM_TensorMap* TM_TensorMap_Create(void)
{
    return new TM_TensorMap{};
}

void TM_TensorMap_Destroy(TM_TensorMap* map)
{
    delete map;
}

void TM_TensorMap_Clear(TM_TensorMap* map)
{
    if (map) {
        map->map.clear();
    }
}

void TM_TensorMap_SetInt32(TM_TensorMap* map, const char* name, const int32_t* data, int ndim, const int64_t* shape)
{
    SetTensorCommon(&map->map, name, data, ndim, shape, turbomind::DeviceType::kCPU);
}

void TM_TensorMap_SetInt64(TM_TensorMap* map, const char* name, const int64_t* data, int ndim, const int64_t* shape)
{
    SetTensorCommon(&map->map, name, data, ndim, shape, turbomind::DeviceType::kCPU);
}

void TM_TensorMap_SetFloat32(TM_TensorMap* map, const char* name, const float* data, int ndim, const int64_t* shape)
{
    SetTensorCommon(&map->map, name, data, ndim, shape, turbomind::DeviceType::kCPU);
}

void TM_TensorMap_SetBytes(TM_TensorMap* map, const char* name, const void* data, size_t size, int ndim, const int64_t* shape)
{
    std::vector<turbomind::core::ssize_t> shape_vec(shape, shape + ndim);
    auto dtype = turbomind::DataType::kUint8;
    auto tensor = turbomind::core::Tensor(
        const_cast<void*>(data),
        turbomind::Layout{shape_vec},
        dtype,
        turbomind::DeviceType::kCPU);
    map->map[name] = std::move(tensor);
}

// ============================================================
// Guided Decoding / Structured Output (xgrammar)
// ============================================================

#include "xgrammar/compiler.h"

namespace {

// Singleton builtin JSON grammar (created once, reused)
TM_CompiledGrammar* g_builtin_json_grammar = nullptr;
std::once_flag g_builtin_json_init_flag;

// Initialize the builtin JSON grammar
void InitBuiltinJSONGrammar() {
    // Stub implementation - xgrammar API requires TokenizerInfo
    try {
        // TODO: Initialize builtin JSON grammar with proper tokenizer info
    }
    catch (const std::exception& e) {
        fprintf(stderr, "[C-API] Failed to initialize builtin JSON grammar: %s\n", e.what());
    }
}

}  // anonymous namespace

struct TM_CompiledGrammar {
    std::shared_ptr<xgrammar::CompiledGrammar> grammar;
};

extern "C" TM_CompiledGrammar* TM_Grammar_CreateFromJSONSchema(const char* json_schema)
{
    if (!json_schema) {
        SetError(TM_ERR_INVALID_ARG, "json_schema must not be NULL");
        return nullptr;
    }
    try {
        // TODO: Create grammar with proper TokenizerInfo
        SetError(TM_ERR_RUNTIME, "xgrammar GrammarCompiler requires TokenizerInfo - not yet implemented");
        return nullptr;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

extern "C" TM_CompiledGrammar* TM_Grammar_CreateFromEBNF(const char* ebnf_string)
{
    if (!ebnf_string) {
        SetError(TM_ERR_INVALID_ARG, "ebnf_string must not be NULL");
        return nullptr;
    }
    try {
        // TODO: Create grammar with proper TokenizerInfo
        SetError(TM_ERR_RUNTIME, "xgrammar GrammarCompiler requires TokenizerInfo - not yet implemented");
        return nullptr;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

extern "C" TM_CompiledGrammar* TM_Grammar_CreateFromRegex(const char* regex)
{
    if (!regex) {
        SetError(TM_ERR_INVALID_ARG, "regex must not be NULL");
        return nullptr;
    }
    try {
        // TODO: Create grammar with proper TokenizerInfo
        SetError(TM_ERR_RUNTIME, "xgrammar GrammarCompiler requires TokenizerInfo - not yet implemented");
        return nullptr;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

extern "C" const TM_CompiledGrammar* TM_Grammar_GetBuiltinJSON(void)
{
    std::call_once(g_builtin_json_init_flag, InitBuiltinJSONGrammar);
    return g_builtin_json_grammar;
}

extern "C" void TM_Grammar_Destroy(TM_CompiledGrammar* grammar)
{
    // Don't destroy the builtin singleton
    if (grammar && grammar != g_builtin_json_grammar) {
        delete grammar;
    }
}

extern "C" int TM_ModelRequest_SetGrammar(TM_ModelRequest* req, const TM_CompiledGrammar* grammar)
{
    // Stub implementation - xgrammar integration requires full headers
    return TM_OK;
}

// ============================================================
// Weight Export / Import
// ============================================================

int TM_ExportWeightsToBin(
    const char* model_path,
    const char* output_dir,
    int data_type,
    int hidden_size,
    int num_layers,
    int num_heads,
    int num_kv_heads,
    int vocab_size)
{
    if (!model_path || !output_dir) {
        SetError(TM_ERR_INVALID_ARG, "model_path and output_dir must not be NULL");
        return TM_ERR_INVALID_ARG;
    }

    try {
        turbomind::WeightSerializeConfig config;
        config.model_dir   = model_path;
        config.output_dir  = output_dir;
        config.data_type   = data_type;
        config.is_awq      = false;  // Auto-detect from config.json
        config.group_size  = 128;
        config.tp_size     = 1;
        config.tp_rank     = 0;
        config.hidden_size = hidden_size;
        config.num_layers  = num_layers;
        config.num_heads   = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.vocab_size  = vocab_size;

        return turbomind::SerializeWeightsToBin(config);
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return TM_ERR_RUNTIME;
    }
}

void TM_TensorMap_SetInt32GPU(TM_TensorMap* map, const char* name, const int32_t* data, int ndim, const int64_t* shape)
{
    SetTensorCommon(&map->map, name, data, ndim, shape, turbomind::DeviceType::kDEVICE);
}

void TM_TensorMap_SetInt64GPU(TM_TensorMap* map, const char* name, const int64_t* data, int ndim, const int64_t* shape)
{
    SetTensorCommon(&map->map, name, data, ndim, shape, turbomind::DeviceType::kDEVICE);
}

void TM_TensorMap_SetFloat32GPU(TM_TensorMap* map, const char* name, const float* data, int ndim, const int64_t* shape)
{
    SetTensorCommon(&map->map, name, data, ndim, shape, turbomind::DeviceType::kDEVICE);
}

// DLPack dtype code -> turbomind::DataType
static turbomind::DataType DLPackDtypeToTurbomind(int dl_type_code, int dl_type_bits)
{
    switch (dl_type_code) {
        case 0: // kDLBool
            return turbomind::DataType::kBool;
        case 2: // kDLInt
            if (dl_type_bits == 8) return turbomind::DataType::kInt8;
            if (dl_type_bits == 16) return turbomind::DataType::kInt16;
            if (dl_type_bits == 32) return turbomind::DataType::kInt32;
            if (dl_type_bits == 64) return turbomind::DataType::kInt64;
            return turbomind::DataType::kNull;
        case 3: // kDLFloat
            if (dl_type_bits == 16) return turbomind::DataType::kHalf;
            if (dl_type_bits == 32) return turbomind::DataType::kFloat32;
            if (dl_type_bits == 64) return turbomind::DataType::kFloat64;
            return turbomind::DataType::kNull;
        case 4: // kDLUInt
            if (dl_type_bits == 8) return turbomind::DataType::kUint8;
            if (dl_type_bits == 16) return turbomind::DataType::kUint16;
            if (dl_type_bits == 32) return turbomind::DataType::kUint32;
            if (dl_type_bits == 64) return turbomind::DataType::kUint64;
            return turbomind::DataType::kNull;
        case 5: // kDLBfloat
            return turbomind::DataType::kBfloat16;
        default:
            return turbomind::DataType::kNull;
    }
}

void TM_TensorMap_SetDLPack(
    TM_TensorMap* map,
    const char* name,
    const void* data,
    int ndim,
    const int64_t* shape,
    int dl_type_code,
    int dl_type_bits,
    int device_type)
{
    turbomind::DataType dtype = DLPackDtypeToTurbomind(dl_type_code, dl_type_bits);
    if (dtype == turbomind::DataType::kNull) {
        // Unsupported dtype, create a uint8 tensor as fallback
        dtype = turbomind::DataType::kUint8;
    }

    turbomind::DeviceType dev = (device_type == 2) ? turbomind::DeviceType::kDEVICE : turbomind::DeviceType::kCPU;
    std::vector<turbomind::core::ssize_t> shape_vec(shape, shape + ndim);
    turbomind::Layout layout{shape_vec};

    auto tensor = turbomind::core::Tensor(
        const_cast<void*>(data),
        layout,
        dtype,
        dev);

    map->map[name] = std::move(tensor);
}

int TM_TensorFromDLPack(void* dlpack_capsule, TM_Tensor* out_tensor)
{
    if (!dlpack_capsule || !out_tensor) {
        return -1;
    }

    const DLManagedTensorVersioned* dmt =
        static_cast<const DLManagedTensorVersioned*>(dlpack_capsule);

    if (dmt->version.major != DLPACK_MAJOR_VERSION) {
        return -1;
    }

    const DLTensor& dt = dmt->dl_tensor;

    out_tensor->dtype = ToCDataType(DLPackDtypeToTurbomind(dt.dtype.code, dt.dtype.bits));
    out_tensor->ndim = dt.ndim;
    out_tensor->data = dt.data;
    out_tensor->device_id = (dt.device.device_type == kDLCUDA || dt.device.device_type == kDLCUDAManaged) ? dt.device.device_id : -1;

    for (int i = 0; i < dt.ndim && i < 8; ++i) {
        out_tensor->shape[i] = dt.shape[i];
    }

    return 0;
}

void* TM_TensorToDLPack(const TM_Tensor* tensor)
{
    if (!tensor || !tensor->data || tensor->ndim <= 0) {
        return nullptr;
    }

    auto dtype = static_cast<turbomind::DataType>(tensor->dtype);

    DLDataType dl_dtype;
    switch (dtype) {
        case turbomind::DataType::kBool:
            dl_dtype = {6, 8, 1}; break;
        case turbomind::DataType::kInt8:
            dl_dtype = {0, 8, 1}; break;
        case turbomind::DataType::kInt16:
            dl_dtype = {0, 16, 1}; break;
        case turbomind::DataType::kInt32:
            dl_dtype = {0, 32, 1}; break;
        case turbomind::DataType::kInt64:
            dl_dtype = {0, 64, 1}; break;
        case turbomind::DataType::kUint8:
            dl_dtype = {1, 8, 1}; break;
        case turbomind::DataType::kUint16:
            dl_dtype = {1, 16, 1}; break;
        case turbomind::DataType::kUint32:
            dl_dtype = {1, 32, 1}; break;
        case turbomind::DataType::kUint64:
            dl_dtype = {1, 64, 1}; break;
        case turbomind::DataType::kHalf:
            dl_dtype = {2, 16, 1}; break;
        case turbomind::DataType::kFloat32:
            dl_dtype = {2, 32, 1}; break;
        case turbomind::DataType::kFloat64:
            dl_dtype = {2, 64, 1}; break;
        case turbomind::DataType::kBfloat16:
            dl_dtype = {4, 16, 1}; break;
        default:
            return nullptr;
    }

    DLDevice dl_device;
    if (tensor->device_id >= 0) {
        dl_device.device_type = kDLCUDA;
        dl_device.device_id = tensor->device_id;
    }
    else {
        dl_device.device_type = kDLCPU;
        dl_device.device_id = 0;
    }

    int64_t* shape = new int64_t[tensor->ndim];
    for (int i = 0; i < tensor->ndim; ++i) {
        shape[i] = tensor->shape[i];
    }

    DLTensor dl_tensor{
        tensor->data,
        dl_device,
        tensor->ndim,
        dl_dtype,
        shape,
        nullptr,
        0
    };

    auto* dmt = new DLManagedTensorVersioned{};
    dmt->version.major = DLPACK_MAJOR_VERSION;
    dmt->version.minor = DLPACK_MINOR_VERSION;
    dmt->dl_tensor = dl_tensor;
    dmt->flags = 0;
    dmt->manager_ctx = shape;
    dmt->deleter = [](DLManagedTensorVersioned* self) {
        delete[] static_cast<int64_t*>(self->manager_ctx);
        delete self;
    };

    return dmt;
}

bool TM_TensorMap_Get(
    TM_TensorMap* map,
    const char* name,
    void** out_data,
    TM_DataType* out_dtype,
    TM_MemoryType* out_memory_type,
    int* out_ndim,
    int64_t* out_shape)
{
    auto it = map->map.find(name);
    if (it == map->map.end()) {
        return false;
    }
    const auto& tensor = it->second;
    if (out_data) *out_data = const_cast<void*>(tensor.raw_data());
    if (out_dtype) *out_dtype = ToCDataType(tensor.dtype());
    if (out_memory_type) *out_memory_type = ToCMemoryType(tensor.device().type);
    if (out_ndim) *out_ndim = static_cast<int>(tensor.shape().size());
    if (out_shape) {
        for (size_t i = 0; i < tensor.shape().size(); ++i) {
            out_shape[i] = tensor.shape()[i];
        }
    }
    return true;
}

// ============================================================
// Generation config
// ============================================================

struct TM_GenerationConfig {
    turbomind::GenerationConfig config;
};

TM_GenerationConfig* TM_GenerationConfig_Create(void)
{
    return new TM_GenerationConfig{};
}

void TM_GenerationConfig_Destroy(TM_GenerationConfig* config)
{
    delete config;
}

void TM_GenerationConfig_SetMaxNewTokens(TM_GenerationConfig* config, int value)
{
    config->config.max_new_tokens = value;
}

void TM_GenerationConfig_SetMinNewTokens(TM_GenerationConfig* config, int value)
{
    config->config.min_new_tokens = value;
}

void TM_GenerationConfig_SetEosIds(TM_GenerationConfig* config, const int* ids, int count)
{
    config->config.eos_ids.assign(ids, ids + count);
}

void TM_GenerationConfig_SetStopIds(TM_GenerationConfig* config, const int* ids, int count)
{
    config->config.stop_ids[0].assign(ids, ids + count);
}

void TM_GenerationConfig_SetBadIds(TM_GenerationConfig* config, const int* ids, int count)
{
    config->config.bad_ids[0].assign(ids, ids + count);
}

void TM_GenerationConfig_SetTopP(TM_GenerationConfig* config, float value)
{
    config->config.top_p = value;
}

void TM_GenerationConfig_SetTopK(TM_GenerationConfig* config, int value)
{
    config->config.top_k = value;
}

void TM_GenerationConfig_SetMinP(TM_GenerationConfig* config, float value)
{
    config->config.min_p = value;
}

void TM_GenerationConfig_SetTemperature(TM_GenerationConfig* config, float value)
{
    config->config.temperature = value;
}

void TM_GenerationConfig_SetRepetitionPenalty(TM_GenerationConfig* config, float value)
{
    config->config.repetition_penalty = value;
}

void TM_GenerationConfig_SetRandomSeed(TM_GenerationConfig* config, uint64_t value)
{
    config->config.random_seed = value;
}

void TM_GenerationConfig_SetOutputLogprobs(TM_GenerationConfig* config, int value)
{
    config->config.output_logprobs = value;
}

void TM_GenerationConfig_SetOutputLastHiddenState(TM_GenerationConfig* config, int value)
{
    config->config.output_last_hidden_state = value;
}

void TM_GenerationConfig_SetOutputLogits(TM_GenerationConfig* config, int value)
{
    config->config.output_logits = value;
}

// ============================================================
// Inference (ModelRequest)
// ============================================================

struct TM_TokenCallbackWrapper {
    TM_TokenCallback func;
    void* user_data;
    int token_id;
    int seq_len;

    TM_TokenCallbackWrapper(TM_TokenCallback f, void* ud)
        : func(f), user_data(ud), token_id(0), seq_len(0) {}
};

struct TM_CompletionCallbackWrapper {
    TM_CompletionCallback func;
    void* user_data;

    TM_CompletionCallbackWrapper(TM_CompletionCallback f, void* ud)
        : func(f), user_data(ud) {}
};

struct TM_ModelRequest {
    turbomind::ModelRequest* req;
    std::shared_ptr<turbomind::TensorMap> output_tensors;
    std::shared_ptr<turbomind::AtomicRequestState> output_state;
    std::shared_ptr<turbomind::RequestMetrics> output_metrics;
    std::shared_ptr<turbomind::TensorMap> streaming_tensors;
    std::shared_ptr<turbomind::AtomicRequestState> streaming_state;
    std::shared_ptr<TM_TokenCallbackWrapper> token_cb_wrapper;
    std::shared_ptr<TM_CompletionCallbackWrapper> completion_cb_wrapper;
};

TM_ModelRequest* TM_ModelRequest_Create(TM_TurboMind* tm)
{
    try {
        auto* mr = new TM_ModelRequest{};
        mr->req = tm->instance->CreateRequest().release();
        return mr;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

void TM_ModelRequest_Destroy(TM_ModelRequest* req)
{
    if (req) {
        delete req->req;
        delete req;
    }
}

int TM_ModelRequest_Forward(
    TM_ModelRequest* req,
    TM_TensorMap* input_tensors,
    const TM_SessionParam* session,
    const TM_GenerationConfig* gen_cfg,
    bool stream_output,
    bool enable_metrics,
    TM_TensorMap* output_tensors)
{
    if (!req || !input_tensors || !session || !gen_cfg || !output_tensors) {
        SetError(TM_ERR_INVALID_ARG, "NULL argument to TM_ModelRequest_Forward");
        return TM_ERR_INVALID_ARG;
    }

    try {
        turbomind::ModelRequest::InputParam param{};
        param.tensors = std::make_shared<turbomind::core::TensorMap>(std::move(input_tensors->map));
        param.session.id = session->id;
        param.session.step = session->step;
        param.session.start_flag = session->start_flag;
        param.session.end_flag = session->end_flag;
        param.gen_cfg = gen_cfg->config;
        param.stream_output = stream_output;
        param.enable_metrics = enable_metrics;

        // Use a promise/future to wait for completion
        auto promise = std::make_shared<std::promise<void>>();
        auto future = promise->get_future();

        auto out = req->req->Forward(std::move(param), [promise]() {
            promise->set_value();
        });

        // Wait for completion
        future.get();

        // Store outputs
        req->output_tensors = std::move(out.tensors);
        req->output_state = std::move(out.state);
        req->output_metrics = std::move(out.metrics);

        // Copy output tensors to caller's TensorMap
        for (const auto& [k, t] : *req->output_tensors) {
            output_tensors->map[k] = t;
        }

        return TM_OK;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return TM_ERR_RUNTIME;
    }
}

int TM_ModelRequest_ForwardAsync(
    TM_ModelRequest* req,
    TM_TensorMap* input_tensors,
    const TM_SessionParam* session,
    const TM_GenerationConfig* gen_cfg,
    bool stream_output,
    bool enable_metrics)
{
    if (!req || !input_tensors || !session || !gen_cfg) {
        SetError(TM_ERR_INVALID_ARG, "NULL argument to TM_ModelRequest_ForwardAsync");
        return TM_ERR_INVALID_ARG;
    }

    try {
        turbomind::ModelRequest::InputParam param{};
        param.tensors = std::make_shared<turbomind::core::TensorMap>(std::move(input_tensors->map));
        param.session.id = session->id;
        param.session.step = session->step;
        param.session.start_flag = session->start_flag;
        param.session.end_flag = session->end_flag;
        param.gen_cfg = gen_cfg->config;
        param.stream_output = stream_output;
        param.enable_metrics = enable_metrics;

        if (req->token_cb_wrapper) {
            param.token_cb = [wrapper = req->token_cb_wrapper](int token_id, int seq_len) {
                wrapper->func(token_id, seq_len, wrapper->user_data);
            };
        }

        // Wire completion callback if set
        bool has_completion_cb = req->completion_cb_wrapper.get() != nullptr;

        // Submit request asynchronously - don't block, let caller poll state
        auto out = req->req->Forward(std::move(param), [has_completion_cb, req]() {
            if (has_completion_cb && req->completion_cb_wrapper) {
                auto state = req->streaming_state ? req->streaming_state->exchange(nullptr) : nullptr;
                auto status = state ? state->status : (int)TM_RequestStatus::TM_STATUS_FINISH;
                auto seq_len = state ? state->seq_len : 0;
                req->completion_cb_wrapper->func(status, seq_len, req->completion_cb_wrapper->user_data);
            }
        });

        // Store outputs as shared state for polling
        req->streaming_tensors = std::move(out.tensors);
        req->streaming_state = std::move(out.state);
        req->output_metrics = std::move(out.metrics);

        return TM_OK;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return TM_ERR_RUNTIME;
    }
}

int TM_ModelRequest_SetTokenCallback(TM_ModelRequest* req, TM_TokenCallback cb, void* user_data)
{
    if (!req) {
        SetError(TM_ERR_INVALID_ARG, "NULL argument to TM_ModelRequest_SetTokenCallback");
        return TM_ERR_INVALID_ARG;
    }

    req->token_cb_wrapper = std::make_shared<TM_TokenCallbackWrapper>(cb, user_data);
    return TM_OK;
}

int TM_ModelRequest_SetCompletionCallback(TM_ModelRequest* req, TM_CompletionCallback cb, void* user_data)
{
    if (!req) {
        SetError(TM_ERR_INVALID_ARG, "NULL argument to TM_ModelRequest_SetCompletionCallback");
        return TM_ERR_INVALID_ARG;
    }

    req->completion_cb_wrapper = std::make_shared<TM_CompletionCallbackWrapper>(cb, user_data);
    return TM_OK;
}

// ============================================================
// CUDA Graph Support (Experimental)
// ============================================================

namespace {

// Internal CUDA Graph structure
struct CudaGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<std::pair<void*, size_t>> buffer_allocations;  // For memory pool management

    ~CudaGraph() {
        if (exec) {
            cudaGraphExecDestroy(exec);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
        // Note: buffer_allocations would need proper cleanup in real implementation
    }
};

}  // anonymous namespace

int TM_CudaGraph_Capture(
    TM_ModelRequest* req,
    TM_TensorMap* input_tensors,
    TM_CudaGraphHandle* out_graph)
{
    if (!req || !input_tensors || !out_graph) {
        SetError(TM_ERR_INVALID_ARG, "NULL argument to TM_CudaGraph_Capture");
        return TM_ERR_INVALID_ARG;
    }

    try {
        // Placeholder: CUDA Graph capture would require:
        // 1. Capturing the graph during a forward pass
        // 2. Instantiating the graph for replay
        // 3. Managing memory pools for deterministic allocation
        //
        // Full implementation requires:
        // - cudaStreamCaptureBegin/End
        // - cudaGraphInstantiate
        // - Custom memory pool for deterministic allocations

        // For now, return not implemented
        SetError(TM_ERR_RUNTIME, "CUDA Graph capture not yet implemented - requires engine-level support");
        return TM_ERR_RUNTIME;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return TM_ERR_RUNTIME;
    }
}

int TM_CudaGraph_Launch(
    TM_CudaGraphHandle graph,
    TM_TensorMap* input_tensors,
    TM_TensorMap* output_tensors,
    void* stream)
{
    if (!graph || !input_tensors || !output_tensors) {
        SetError(TM_ERR_INVALID_ARG, "NULL argument to TM_CudaGraph_Launch");
        return TM_ERR_INVALID_ARG;
    }

    try {
        // Placeholder: CUDA Graph launch would:
        // 1. Update input tensor pointers in the graph
        // 2. cudaGraphLaunch with the appropriate stream
        // 3. Handle output tensor extraction

        SetError(TM_ERR_RUNTIME, "CUDA Graph launch not yet implemented");
        return TM_ERR_RUNTIME;
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return TM_ERR_RUNTIME;
    }
}

void TM_CudaGraph_Destroy(TM_CudaGraphHandle graph)
{
    if (graph) {
        delete static_cast<CudaGraph*>(graph);
    }
}

int TM_ModelRequest_GetStreamToken(
    TM_ModelRequest* req,
    void** out_data,
    size_t* out_count)
{
    if (!req || !out_data || !out_count) {
        SetError(TM_ERR_INVALID_ARG, "NULL argument to TM_ModelRequest_GetStreamToken");
        return TM_ERR_INVALID_ARG;
    }

    if (!req->streaming_tensors) {
        SetError(TM_ERR_RUNTIME, "No streaming request in progress");
        return TM_ERR_RUNTIME;
    }

    auto it = req->streaming_tensors->find("output_ids");
    if (it == req->streaming_tensors->end()) {
        SetError(TM_ERR_NOT_FOUND, "output_ids not found in streaming tensors");
        return TM_ERR_NOT_FOUND;
    }

    const auto& tensor = it->second;
    *out_data = const_cast<void*>(tensor.raw_data());
    *out_count = static_cast<size_t>(tensor.shape(0));

    return 0;
}

void TM_ModelRequest_Cancel(TM_ModelRequest* req)
{
    req->req->Cancel();
}

void TM_ModelRequest_End(TM_ModelRequest* req, uint64_t session_id)
{
    req->req->End([](int) {}, session_id);
}

int TM_ModelRequest_GetState(TM_ModelRequest* req, TM_RequestStatus* out_status, int* out_seq_len)
{
    if (!req || !out_status) {
        return TM_ERR_INVALID_ARG;
    }

    auto state = req->output_state ? req->output_state->exchange(nullptr) : nullptr;
    if (!state) {
        // No state available - request may not have completed yet
        *out_status = TM_STATUS_OK;
        if (out_seq_len) *out_seq_len = 0;
        return 0;
    }

    *out_status = static_cast<TM_RequestStatus>(state->status);
    if (out_seq_len) *out_seq_len = state->seq_len;
    return 0;
}

int TM_ModelRequest_GetStreamingState(TM_ModelRequest* req, TM_RequestStatus* out_status, int* out_seq_len)
{
    if (!req || !out_status) {
        return TM_ERR_INVALID_ARG;
    }

    auto state = req->streaming_state ? req->streaming_state->exchange(nullptr) : nullptr;
    if (!state) {
        // No state available - request still running or not started
        *out_status = TM_STATUS_OK;
        if (out_seq_len) *out_seq_len = 0;
        return 0;
    }

    *out_status = static_cast<TM_RequestStatus>(state->status);
    if (out_seq_len) *out_seq_len = state->seq_len;
    return 0;
}

int TM_ModelRequest_GetOutput(
    TM_ModelRequest* req,
    const char* name,
    void** out_data,
    size_t* out_size)
{
    if (!req || !name || !out_data || !out_size) {
        return TM_ERR_INVALID_ARG;
    }

    if (!req->output_tensors) {
        SetError(TM_ERR_RUNTIME, "Request has not been forwarded yet");
        return TM_ERR_RUNTIME;
    }

    auto it = req->output_tensors->find(name);
    if (it == req->output_tensors->end()) {
        return TM_ERR_NOT_FOUND;
    }

    const auto& tensor = it->second;
    *out_data = const_cast<void*>(tensor.raw_data());
    *out_size = static_cast<size_t>(tensor.size());
    return 0;
}