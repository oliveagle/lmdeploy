// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <string>
#include <vector>

namespace turbomind {

// Forward declarations
class Module;

namespace core {

// Forward declaration for core::Module
class Module;

}  // namespace core

/// Weight file metadata for serialization
struct WeightFileInfo {
    std::string name;       // Weight name (e.g., "tok_embeddings", "layers.0.attention.w_qkv")
    std::string file_path;  // Path to the .bin file
    size_t      offset;     // Offset within the file
    size_t      size;       // Size in bytes
    int         dtype;      // Data type (DataType enum value)
    std::vector<int> shape; // Tensor shape
};

/// Configuration for weight serialization
struct WeightSerializeConfig {
    std::string model_dir;          // Source model directory (HF safetensors)
    std::string output_dir;         // Output directory for .bin files
    int         data_type;          // Target data type (FP16/BF16)
    bool        is_awq;             // AWQ quantized model
    int         group_size;         // AWQ group size (128 for AWQ)
    int         tp_size;            // Tensor parallel size
    int         tp_rank;            // Tensor parallel rank
    int         hidden_size;        // Model hidden size
    int         num_layers;         // Number of layers
    int         num_heads;          // Number of attention heads
    int         num_kv_heads;       // Number of KV heads (for GQA)
    int         vocab_size;         // Vocabulary size
};

/// Serialize weights from HF safetensors to TurboMind .bin format
///
/// This function reads HuggingFace safetensors files and converts them
/// to the TurboMind .bin file format that can be loaded by InitFromPath.
///
/// Args:
///     config: Serialization configuration
///
/// Returns:
///     0 on success, non-zero on failure
///
/// The function creates:
/// 1. config.yaml - TurboMind configuration file
/// 2. *.bin files - One or more binary weight files
/// 3. weight_index.json - Index mapping weight names to file locations
int SerializeWeightsToBin(const WeightSerializeConfig& config);

/// Load weights from .bin files into a ModelWeight module
///
/// This is the counterpart to SerializeWeightsToBin() and is used
/// by InitFromPath() to load weights from disk.
///
/// Args:
///     model_dir: Directory containing config.yaml and .bin files
///     weight_module: ModelWeight module to populate
///
/// Returns:
///     0 on success, non-zero on failure
int LoadWeightsFromBin(const std::string& model_dir, void* weight_module);

}  // namespace turbomind
