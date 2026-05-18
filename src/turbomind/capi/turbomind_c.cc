// Copyright (c) OpenMMLab. All rights reserved.
// C API implementation for TurboMind inference engine

#include "turbomind_c.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/turbomind/core/module.h"
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
        auto* tm = new TM_TurboMind{};

        // Create a simple GIL factory (no-op for C API)
        auto gil_factory = []() -> std::shared_ptr<void> { return nullptr; };

        tm->instance = std::make_unique<turbomind::TurboMind>(
            model_dir,
            std::move(config->config),
            std::move(gil_factory));

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
    config.intermediate_size = get_int("intermediate_size", config.hidden_size * 4);

    // Model type
    config.model_type = get_string("model_type", "llama");
    config.arch = get_string("arch", config.arch);

    // MoE configuration
    config.num_local_experts = get_int("num_local_experts", 0);
    config.num_experts_per_tok = get_int("num_experts_per_tok", 0);

    // DeltaNet / Linear Attention
    config.use_linear_attn = get_bool("use_linear_attn", false);

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
static std::string MapHuggingFaceWeightToTurboMind(const std::string& hf_name);
static void LoadWeightsFromSafetensors(
    turbomind::ModelWeight* model_weight,
    const char* safetensors_path,
    const HfModelConfig& hf_config);

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
static void LoadWeightsFromSafetensors(
    turbomind::ModelWeight* model_weight,
    const char* safetensors_path,
    const HfModelConfig& hf_config)
{
    try {
        turbomind::SafetensorsReader reader(safetensors_path);

        // Iterate over all tensors in the file
        for (size_t i = 0; i < reader.num_tensors(); ++i) {
            const std::string& tensor_name = reader.tensor_name(i);

            // Map HF weight names to TurboMind module paths
            // HF format: model.layers.0.self_attn.q_proj.weight
            // TM format: layers.0.attention.w_qkv.weight

            std::string tm_path = MapHuggingFaceWeightToTurboMind(tensor_name);
            if (tm_path.empty()) {
                continue;  // Skip unmapped weights
            }

            // Get tensor metadata
            const auto* meta = reader.get_tensor_meta(tensor_name);
            if (!meta) {
                continue;  // Skip failed reads
            }

            // Read tensor data
            std::vector<uint8_t> data = reader.read_tensor(tensor_name);
            if (data.empty()) {
                continue;  // Skip failed reads
            }

            // Parse the TurboMind path to find the module and param
            // Format: "layers.0.attention.w_qkv.weight"
            std::vector<std::string> parts;
            std::stringstream ss(tm_path);
            std::string part;
            while (std::getline(ss, part, '.')) {
                parts.push_back(part);
            }

            if (parts.empty()) {
                continue;
            }

            // Navigate to the target module
            turbomind::core::Module* current = model_weight;
            for (size_t j = 0; j < parts.size() - 1; ++j) {
                if (!current) break;
                current = current->child(parts[j]);
            }

            if (!current) {
                continue;
            }

            // The last part is the param name (e.g., "weight")
            std::string param_name = parts.back();

            // Allocate and copy the tensor
            auto param = current->param(param_name);
            if (param) {
                // Convert shape from size_t to int64_t for compatibility
                std::vector<int64_t> shape64;
                for (const auto& dim : meta->shape) {
                    shape64.push_back(static_cast<int64_t>(dim));
                }

                turbomind::DataType tm_dtype = meta->dtype;

                param.alloc(meta->shape, tm_dtype);

                // Copy data (TODO: handle device placement)
                auto tensor = param.get();
                if (tensor && tensor.raw_data()) {
                    size_t copy_size = std::min(data.size(), static_cast<size_t>(tensor.byte_size()));
                    std::memcpy(tensor.raw_data(), data.data(), copy_size);
                }
            }
        }
    }
    catch (const std::exception& e) {
        // Log error but continue with other files
        std::cerr << "Error loading safetensors file " << safetensors_path << ": " << e.what() << std::endl;
    }
}

// Map HuggingFace weight names to TurboMind module paths
static std::string MapHuggingFaceWeightToTurboMind(const std::string& hf_name)
{
    // HF format: model.layers.0.self_attn.q_proj.weight
    // TM format: layers.0.attention.w_qkv.weight

    std::string result = hf_name;

    // Remove "model." prefix if present
    if (result.find("model.") == 0) {
        result = result.substr(6);
    }

    // Remove ".language_model." prefix if present (for vision-language models)
    size_t lang_pos = result.find(".language_model.");
    if (lang_pos != std::string::npos) {
        result = result.substr(0, lang_pos) + result.substr(lang_pos + 17);
    }

    // Replace common patterns
    // self_attn -> attention
    size_t self_attn_pos = result.find(".self_attn.");
    while (self_attn_pos != std::string::npos) {
        result.replace(self_attn_pos, 11, ".attention.");
        self_attn_pos = result.find(".self_attn.");
    }

    // mlp -> feed_forward
    size_t mlp_pos = result.find(".mlp.");
    while (mlp_pos != std::string::npos) {
        result.replace(mlp_pos, 5, ".feed_forward.");
        mlp_pos = result.find(".mlp.");
    }

    // input_layernorm -> attention_norm
    size_t input_ln_pos = result.find(".input_layernorm");
    while (input_ln_pos != std::string::npos) {
        result.replace(input_ln_pos, 16, ".attention_norm");
        input_ln_pos = result.find(".input_layernorm");
    }

    // post_attention_layernorm -> ffn_norm
    size_t post_ln_pos = result.find(".post_attention_layernorm");
    while (post_ln_pos != std::string::npos) {
        result.replace(post_ln_pos, 24, ".ffn_norm");
        post_ln_pos = result.find(".post_attention_layernorm");
    }

    // QKV projection handling
    // q_proj, k_proj, v_proj -> w_qkv (fused)
    // This is complex because we need to fuse multiple tensors
    // For now, just map individual names
    size_t q_proj_pos = result.find(".q_proj.");
    if (q_proj_pos != std::string::npos) {
        result.replace(q_proj_pos, 9, ".q_proj.");
    }

    size_t k_proj_pos = result.find(".k_proj.");
    if (k_proj_pos != std::string::npos) {
        result.replace(k_proj_pos, 9, ".k_proj.");
    }

    size_t v_proj_pos = result.find(".v_proj.");
    if (v_proj_pos != std::string::npos) {
        result.replace(v_proj_pos, 9, ".v_proj.");
    }

    size_t o_proj_pos = result.find(".o_proj.");
    if (o_proj_pos != std::string::npos) {
        result.replace(o_proj_pos, 9, ".wo.");
    }

    // FFN layers
    size_t gate_proj_pos = result.find(".gate_proj.");
    if (gate_proj_pos != std::string::npos) {
        result.replace(gate_proj_pos, 12, ".w1.");
    }

    size_t up_proj_pos = result.find(".up_proj.");
    if (up_proj_pos != std::string::npos) {
        result.replace(up_proj_pos, 9, ".w3.");
    }

    size_t down_proj_pos = result.find(".down_proj.");
    if (down_proj_pos != std::string::npos) {
        result.replace(down_proj_pos, 11, ".w2.");
    }

    // AWQ quantization parameters
    // weight_scale -> scales, weight_zero -> zeros
    size_t scale_pos = result.find(".weight_scale");
    while (scale_pos != std::string::npos) {
        result.replace(scale_pos, 13, ".scales");
        scale_pos = result.find(".weight_scale");
    }

    size_t zero_pos = result.find(".weight_zero");
    while (zero_pos != std::string::npos) {
        result.replace(zero_pos, 12, ".zeros");
        zero_pos = result.find(".weight_zero");
    }

    // Embeddings
    size_t embed_pos = result.find(".embed_tokens.");
    if (embed_pos != std::string::npos) {
        result.replace(embed_pos, 14, ".tok_embeddings.");
    }

    size_t lm_head_pos = result.find(".lm_head.");
    if (lm_head_pos != std::string::npos) {
        result.replace(lm_head_pos, 9, ".output.");
    }

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
    if (!tm || !model_dir) {
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

        // Step 3: Parse HuggingFace config.json
        HfModelConfig hf_config = ParseHfConfig(std::string(model_dir));

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

        // Build the complete module tree
        auto* model_weight = static_cast<turbomind::ModelWeight*>(weight_module.get());

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

                // Create q_proj LinearWeight child
                auto q_proj_cfg = CreateAwqLinearConfig(
                    hf_config.hidden_size, hf_config.num_attention_heads * head_dim,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto q_proj_module = turbomind::core::Module::create(q_proj_cfg);
                if (q_proj_module) {
                    attn->add_child("q_proj", std::move(q_proj_module));
                }

                // Create k_proj LinearWeight child
                auto k_proj_cfg = CreateAwqLinearConfig(
                    hf_config.hidden_size, hf_config.num_key_value_heads * head_dim,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto k_proj_module = turbomind::core::Module::create(k_proj_cfg);
                if (k_proj_module) {
                    attn->add_child("k_proj", std::move(k_proj_module));
                }

                // Create v_proj LinearWeight child
                auto v_proj_cfg = CreateAwqLinearConfig(
                    hf_config.hidden_size, hf_config.num_key_value_heads * head_dim,
                    weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                auto v_proj_module = turbomind::core::Module::create(v_proj_cfg);
                if (v_proj_module) {
                    attn->add_child("v_proj", std::move(v_proj_module));
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

            // 4f. Create linear_attn (DeltaNetWeight) with its child modules if this model has linear attention
            if (hf_config.use_linear_attn) {
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

                    // Create in_proj_all LinearWeight child
                    auto in_proj_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, 3 * hf_config.hidden_size,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto in_proj_module = turbomind::core::Module::create(in_proj_cfg);
                    if (in_proj_module) {
                        delta->add_child("in_proj_all", std::move(in_proj_module));
                    }

                    // Create out_proj LinearWeight child
                    auto out_proj_cfg = CreateAwqLinearConfig(
                        hf_config.hidden_size, hf_config.hidden_size,
                        weight_cfg.data_type, hf_config.is_awq, hf_config.awq_group_size);
                    auto out_proj_module = turbomind::core::Module::create(out_proj_cfg);
                    if (out_proj_module) {
                        delta->add_child("out_proj", std::move(out_proj_module));
                    }

                    // Create norm NormWeight child
                    turbomind::core::NormConfig delta_norm_cfg;
                    delta_norm_cfg.dim = hf_config.hidden_size;
                    delta_norm_cfg.data_type = weight_cfg.data_type;
                    delta_norm_cfg.norm_eps = 1e-6f;
                    auto delta_norm_module = turbomind::core::Module::create(delta_norm_cfg);
                    if (delta_norm_module) {
                        delta->add_child("norm", std::move(delta_norm_module));
                    }

                    decoder_layer->add_child("linear_attn", std::move(delta_module));
                }
            }

            // Add layer to ModuleList
            layers_list->add_child(std::to_string(layer_idx), std::move(layer_module));
        }

        // Add layers to ModelWeight
        model_weight->add_child("layers", std::move(layers_list_unique));

        // Attach to ModelRoot via add_child
        auto* result = root->add_child("text_model", std::move(weight_module));
        if (!result) {
            SetError(TM_ERR_RUNTIME, "Failed to attach ModelWeight to ModelRoot");
            return TM_ERR_RUNTIME;
        }

        // Step 4: Load weights from safetensors files
        // Find all safetensors files in the model directory
        std::vector<std::string> safetensors_files = FindSafetensorsFiles(model_dir);

        if (safetensors_files.empty()) {
            SetError(TM_ERR_RUNTIME, "No safetensors files found in model directory");
            return TM_ERR_RUNTIME;
        }

        // Load weights from each safetensors file
        for (const auto& st_file : safetensors_files) {
            LoadWeightsFromSafetensors(model_weight, st_file.c_str(), hf_config);
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

struct TM_ModelRequest {
    turbomind::ModelRequest* req;
    std::shared_ptr<turbomind::TensorMap> output_tensors;
    std::shared_ptr<turbomind::AtomicRequestState> output_state;
    std::shared_ptr<turbomind::RequestMetrics> output_metrics;
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