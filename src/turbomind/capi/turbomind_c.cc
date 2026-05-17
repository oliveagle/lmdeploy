// Copyright (c) OpenMMLab. All rights reserved.
// C API implementation for TurboMind inference engine

#include "turbomind_c.h"

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
#include "src/turbomind/turbomind.h"

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

// Helper to read config.json and extract hidden_size
// Returns hidden_size on success, -1 on failure
int ReadHiddenSizeFromConfig(const std::string& model_dir)
{
    std::string config_path = model_dir;
    if (!config_path.empty() && config_path.back() != '/') {
        config_path += '/';
    }
    config_path += "config.json";

    std::ifstream f(config_path);
    if (!f.is_open()) {
        return -1;
    }

    // Simple JSON parser to extract "hidden_size"
    std::string line;
    while (std::getline(f, line)) {
        // Look for "hidden_size": <number>
        size_t pos = line.find("\"hidden_size\"");
        if (pos != std::string::npos) {
            size_t colon = line.find(':', pos);
            if (colon != std::string::npos) {
                size_t start = colon + 1;
                // Skip whitespace
                while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
                    ++start;
                }
                // Extract number
                size_t end = start;
                while (end < line.size() && (isdigit(line[end]) || line[end] == '-')) {
                    ++end;
                }
                if (end > start) {
                    return std::stoi(line.substr(start, end - start));
                }
            }
        }
    }
    return -1;
}

// AWQ Quantization configuration
struct AwqQuantConfig {
    bool     is_enabled = false;
    int      bits = 4;
    int      group_size = 128;
    std::string quant_method = "awq";
    std::string version = "gemm";
    bool     symmetric = true;
    bool     zero_point = true;
    bool     pack = true;
};

// Helper to read quantization_config from config.json
// Returns AwqQuantConfig with is_enabled=true if AWQ quantization is detected
AwqQuantConfig ReadAwqQuantConfig(const std::string& model_dir)
{
    AwqQuantConfig config;
    std::string config_path = model_dir;
    if (!config_path.empty() && config_path.back() != '/') {
        config_path += '/';
    }
    config_path += "config.json";

    std::ifstream f(config_path);
    if (!f.is_open()) {
        return config;  // is_enabled = false
    }

    // Read entire file into string for easier parsing
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Look for quantization_config block
    size_t quant_start = content.find("\"quantization_config\":");
    if (quant_start == std::string::npos) {
        return config;  // No quantization config
    }

    // Find the opening brace of quantization_config
    size_t obj_start = content.find('{', quant_start);
    if (obj_start == std::string::npos) {
        return config;
    }

    // Find matching closing brace (simple depth counter)
    int depth = 1;
    size_t obj_end = obj_start + 1;
    while (obj_end < content.size() && depth > 0) {
        if (content[obj_end] == '{') depth++;
        else if (content[obj_end] == '}') depth--;
        obj_end++;
    }

    if (depth != 0) {
        return config;  // Malformed JSON
    }

    std::string quant_json = content.substr(obj_start, obj_end - obj_start);

    // Check quantization method
    size_t method_pos = quant_json.find("\"quant_method\":");
    if (method_pos != std::string::npos) {
        size_t colon = quant_json.find(':', method_pos);
        size_t quote_start = quant_json.find('"', colon);
        if (quote_start != std::string::npos) {
            size_t quote_end = quant_json.find('"', quote_start + 1);
            if (quote_end != std::string::npos) {
                std::string method = quant_json.substr(quote_start + 1, quote_end - quote_start - 1);
                if (method == "awq") {
                    config.is_enabled = true;
                    config.quant_method = method;
                }
            }
        }
    }

    if (!config.is_enabled) {
        return config;
    }

    // Parse AWQ parameters
    auto parse_int = [&](const std::string& key, int default_val) -> int {
        size_t pos = quant_json.find("\"" + key + "\":");
        if (pos != std::string::npos) {
            size_t colon = quant_json.find(':', pos);
            size_t start = colon + 1;
            while (start < quant_json.size() && (quant_json[start] == ' ' || quant_json[start] == '\t' || quant_json[start] == '\n')) {
                start++;
            }
            size_t end = start;
            while (end < quant_json.size() && (isdigit(quant_json[end]) || quant_json[end] == '-')) {
                end++;
            }
            if (end > start) {
                return std::stoi(quant_json.substr(start, end - start));
            }
        }
        return default_val;
    };

    auto parse_bool = [&](const std::string& key, bool default_val) -> bool {
        size_t pos = quant_json.find("\"" + key + "\":");
        if (pos != std::string::npos) {
            size_t colon = quant_json.find(':', pos);
            size_t start = colon + 1;
            while (start < quant_json.size() && (quant_json[start] == ' ' || quant_json[start] == '\t' || quant_json[start] == '\n')) {
                start++;
            }
            size_t end = start;
            while (end < quant_json.size() && (quant_json[end] != ',' && quant_json[end] != '}')) {
                end++;
            }
            std::string value = quant_json.substr(start, end - start);
            // Trim whitespace
            size_t value_start = 0;
            size_t value_end = value.size();
            while (value_start < value_end && (value[value_start] == ' ' || value[value_start] == '\t' || value[value_start] == '\n')) {
                value_start++;
            }
            while (value_end > value_start && (value[value_end - 1] == ' ' || value[value_end - 1] == '\t' || value[value_end - 1] == '\n')) {
                value_end--;
            }
            value = value.substr(value_start, value_end - value_start);
            return value == "true";
        }
        return default_val;
    };

    auto parse_string = [&](const std::string& key, const std::string& default_val) -> std::string {
        size_t pos = quant_json.find("\"" + key + "\":");
        if (pos != std::string::npos) {
            size_t colon = quant_json.find(':', pos);
            size_t quote_start = quant_json.find('"', colon);
            if (quote_start != std::string::npos) {
                size_t quote_end = quant_json.find('"', quote_start + 1);
                if (quote_end != std::string::npos) {
                    return quant_json.substr(quote_start + 1, quote_end - quote_start - 1);
                }
            }
        }
        return default_val;
    };

    config.bits = parse_int("bits", 4);
    config.group_size = parse_int("group_size", 128);
    config.version = parse_string("version", "gemm");
    config.symmetric = parse_bool("symmetric", true);
    config.zero_point = parse_bool("zero_point", true);
    config.pack = parse_bool("pack", true);

    return config;
}

}  // anonymous namespace

// ============================================================
// Safetensors file handling (simple implementation)
// ============================================================

// #include <safetensors.h>  // Optional: use if available in _deps

namespace {

// Simple safetensors reader that doesn't depend on external library
// Format: header (JSON) + tensor data

struct SafetensorsHeader {
    std::vector<std::string> names;
    std::vector<std::vector<size_t>> shapes;
    std::vector<size_t> offsets;
    std::vector<size_t> sizes;
    TM_DataType dtype;
};

struct SafetensorsReader {
    std::string file_path;
    std::ifstream stream;
    size_t header_size;
    std::vector<std::string> names;
    std::vector<std::vector<size_t>> shapes;
    std::vector<size_t> offsets;
    std::vector<size_t> sizes;
    TM_DataType dtype;

    SafetensorsReader(const char* path) : file_path(path), stream(path, std::ios::binary), dtype(TM_DATATYPE_FP32) {
        if (!stream.is_open()) {
            throw std::runtime_error("Cannot open file");
        }

        // Read 8-byte header size (little-endian)
        uint8_t size_bytes[8];
        stream.read(reinterpret_cast<char*>(size_bytes), 8);
        header_size = 0;
        for (int i = 0; i < 8; ++i) {
            header_size |= static_cast<size_t>(size_bytes[i]) << (i * 8);
        }

        // Read header JSON
        std::vector<char> header_json(header_size);
        stream.read(header_json.data(), header_size);

        // Parse JSON to extract tensor metadata
        std::string json_str(header_json.begin(), header_json.end());
        ParseHeader(json_str);
    }

    void ParseHeader(const std::string& json) {
        // Simple JSON parsing for safetensors format
        // {"tensor_name": {"dtype": "F32", "shape": [1, 768], "data_offsets": [0, 3072]}}
        size_t pos = 0;
        while (pos < json.size()) {
            // Find tensor name
            size_t name_start = json.find('"', pos);
            if (name_start == std::string::npos) break;
            name_start += 1;
            size_t name_end = json.find('"', name_start);
            if (name_end == std::string::npos) break;

            std::string tensor_name = json.substr(name_start, name_end - name_start);
            names.push_back(tensor_name);

            // Find dtype
            size_t dtype_pos = json.find("\"dtype\"", name_end);
            if (dtype_pos != std::string::npos) {
                size_t dtype_start = json.find('"', dtype_pos + 6);
                size_t dtype_end = json.find('"', dtype_start + 1);
                std::string dtype_str = json.substr(dtype_start + 1, dtype_end - dtype_start - 1);

                if (dtype_str == "F32") dtype = TM_DATATYPE_FP32;
                else if (dtype_str == "F16" || dtype_str == "fp16") dtype = TM_DATATYPE_FP16;
                else if (dtype_str == "BF16") dtype = TM_DATATYPE_BF16;
                else if (dtype_str == "I64") dtype = TM_DATATYPE_INT64;
                else if (dtype_str == "I32") dtype = TM_DATATYPE_INT32;
                else if (dtype_str == "U8") dtype = TM_DATATYPE_UINT8;
                else dtype = TM_DATATYPE_FP32;
            }

            // Find shape
            std::vector<size_t> shape;
            size_t shape_pos = json.find("\"shape\"", name_end);
            if (shape_pos != std::string::npos) {
                size_t bracket = json.find('[', shape_pos);
                size_t bracket_end = json.find(']', bracket);
                std::string shape_str = json.substr(bracket + 1, bracket_end - bracket - 1);

                size_t num_start = 0;
                while (num_start < shape_str.size()) {
                    size_t comma = shape_str.find(',', num_start);
                    if (comma == std::string::npos) comma = shape_str.size();
                    std::string num_str = shape_str.substr(num_start, comma - num_start);
                    // Trim whitespace
                    size_t first = num_str.find_first_not_of(" \t\n\r");
                    size_t last = num_str.find_last_not_of(" \t\n\r");
                    if (first != std::string::npos) {
                        shape.push_back(std::stoll(num_str.substr(first, last - first + 1)));
                    }
                    num_start = comma + 1;
                }
            }
            shapes.push_back(shape);

            // Find data_offsets
            size_t offsets_pos = json.find("\"data_offsets\"", name_end);
            std::vector<size_t> offsets;
            if (offsets_pos != std::string::npos) {
                size_t bracket = json.find('[', offsets_pos);
                size_t bracket_end = json.find(']', bracket);
                std::string offsets_str = json.substr(bracket + 1, bracket_end - bracket - 1);

                size_t num_start = 0;
                while (num_start < offsets_str.size()) {
                    size_t comma = std::min(offsets_str.find(',', num_start), offsets_str.size());
                    std::string num_str = offsets_str.substr(num_start, comma - num_start);
                    size_t first = num_str.find_first_not_of(" \t\n\r");
                    size_t last = num_str.find_last_not_of(" \t\n\r");
                    if (first != std::string::npos) {
                        offsets.push_back(std::stoll(num_str.substr(first, last - first + 1)));
                    }
                    if (comma == offsets_str.size()) break;
                    num_start = comma + 1;
                }
            }
            if (offsets.size() >= 2) {
                this->offsets.push_back(offsets[0]);
                this->sizes.push_back(offsets[1] - offsets[0]);
            }

            pos = json.find('{', name_end);
        }
    }

    size_t GetDataOffset(const std::string& name) const {
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == name) {
                return offsets[i];
            }
        }
        return 0;
    }

    size_t GetDataSize(const std::string& name) const {
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == name) {
                return sizes[i];
            }
        }
        return 0;
    }
};

}  // anonymous namespace

void* TM_Safetensors_Open(const char* file_path)
{
    try {
        return new SafetensorsReader(file_path);
    }
    catch (const std::exception& e) {
        SetError(TM_ERR_RUNTIME, e.what());
        return nullptr;
    }
}

void TM_Safetensors_Close(void* handle)
{
    delete static_cast<SafetensorsReader*>(handle);
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

    auto* reader = static_cast<SafetensorsReader*>(handle);

    // Find tensor index
    int tensor_idx = -1;
    for (int i = 0; i < static_cast<int>(reader->names.size()); ++i) {
        if (reader->names[i] == name) {
            tensor_idx = i;
            break;
        }
    }

    if (tensor_idx < 0) {
        SetError(TM_ERR_NOT_FOUND, "Tensor not found");
        return -1;
    }

    // Calculate offset: 8-byte header + header_size + tensor_data_offset
    size_t data_offset = 8 + reader->header_size + reader->offsets[tensor_idx];
    size_t data_size = reader->sizes[tensor_idx];

    // Seek to tensor data
    reader->stream.seekg(static_cast<std::streampos>(data_offset));
    if (!reader->stream.good()) {
        SetError(TM_ERR_RUNTIME, "Failed to seek to tensor data");
        return -1;
    }

    // Allocate buffer and read
    std::vector<char> buffer(data_size);
    reader->stream.read(buffer.data(), data_size);
    if (!reader->stream.good()) {
        SetError(TM_ERR_RUNTIME, "Failed to read tensor data");
        return -1;
    }

    // Copy to new buffer (caller takes ownership)
    char* data_copy = new char[data_size];
    std::memcpy(data_copy, buffer.data(), data_size);
    *out_data = data_copy;
    *out_size = data_size;

    // Return shape info
    const auto& shape = reader->shapes[tensor_idx];
    *out_ndim = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        out_shape[i] = static_cast<int64_t>(shape[i]);
    }
    *out_dtype = reader->dtype;

    return 0;
}

int TM_Safetensors_NumTensors(void* handle)
{
    if (!handle) return 0;
    auto* reader = static_cast<SafetensorsReader*>(handle);
    return static_cast<int>(reader->names.size());
}

const char* TM_Safetensors_GetTensorName(void* handle, int index)
{
    if (!handle) return nullptr;
    auto* reader = static_cast<SafetensorsReader*>(handle);
    if (index < 0 || index >= static_cast<int>(reader->names.size())) return nullptr;
    // Note: returned pointer is only valid until SafetensorsReader is destroyed
    static thread_local std::string last_name;
    last_name = reader->names[index];
    return last_name.c_str();
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

        // Step 3: Build and attach ModelWeight
        // Read config.json to get hidden_size and AWQ quantization config
        int hidden_size = ReadHiddenSizeFromConfig(model_dir);
        if (hidden_size <= 0) {
            // Fallback to a reasonable default if config.json is not found or doesn't contain hidden_size
            // This will be overridden during weight loading if needed
            hidden_size = 4096;
        }

        // Read AWQ quantization configuration
        AwqQuantConfig awq_config = ReadAwqQuantConfig(model_dir);

        // Create ModelWeightConfig with appropriate settings
        turbomind::core::ModelWeightConfig weight_cfg;
        weight_cfg.hidden_units = hidden_size;
        weight_cfg.tp_size = 1;
        weight_cfg.tp_rank = 0;

        // Determine data type based on quantization
        if (awq_config.is_enabled) {
            // AWQ 4-bit quantized models use INT8 or FP16 for activations
            // The actual quantized weights (INT4) are handled separately
            weight_cfg.data_type = turbomind::DataType::kFloat16;
        } else {
            weight_cfg.data_type = turbomind::DataType::kFloat16;  // Default to FP16
        }

        // Create ModelWeight via Module::create
        auto weight_module = turbomind::core::Module::create(weight_cfg);
        if (!weight_module) {
            SetError(TM_ERR_RUNTIME, "Failed to create ModelWeight module");
            return TM_ERR_RUNTIME;
        }

        // Attach to ModelRoot via add_child
        auto* result = root->add_child("text_model", std::move(weight_module));
        if (!result) {
            SetError(TM_ERR_RUNTIME, "Failed to attach ModelWeight to ModelRoot");
            return TM_ERR_RUNTIME;
        }

        // Step 4: Process weights (moves weights to GPU and calls prepare)
        // AWQ quantization: weights are loaded from safetensors with scales/zeros
        // The ProcessWeights step handles dequantization during weight loading
        tm->instance->ProcessWeights(index);

        // Step 5: Create inference engine
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