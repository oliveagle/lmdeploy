// Copyright (c) OpenMMLab. All rights reserved.
// Header-only safetensors file reader for TurboMind
// Format: 8-byte header (little-endian) + JSON metadata + binary tensor data

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/turbomind/core/data_type.h"

namespace turbomind {

/// Safetensors file reader (header-only implementation)
///
/// Usage:
///   auto reader = SafetensorsReader("/path/to/model.safetensors");
///   for (size_t i = 0; i < reader.num_tensors(); ++i) {
///       auto tensor = reader.read_tensor(reader.tensor_name(i));
///       // process tensor...
///   }
class SafetensorsReader {
public:
    /// Tensor metadata from safetensors header
    struct TensorMeta {
        std::string name;
        std::vector<size_t> shape;
        size_t offset;       // Offset within the binary blob
        size_t size;         // Size in bytes
        DataType dtype;      // TurboMind data type
    };

    /// Open a safetensors file and parse the header
    explicit SafetensorsReader(const char* file_path) : stream_(file_path, std::ios::binary)
    {
        if (!stream_.is_open()) {
            throw std::runtime_error(std::string("Cannot open safetensors file: ") + file_path);
        }
        ParseHeader();
    }

    explicit SafetensorsReader(const std::string& file_path) : SafetensorsReader(file_path.c_str()) {}

    ~SafetensorsReader() = default;

    // Non-copyable, movable
    SafetensorsReader(const SafetensorsReader&) = delete;
    SafetensorsReader& operator=(const SafetensorsReader&) = delete;
    SafetensorsReader(SafetensorsReader&&) = default;
    SafetensorsReader& operator=(SafetensorsReader&&) = default;

    /// Number of tensors in this file
    size_t num_tensors() const { return metas_.size(); }

    /// Get tensor name by index
    const std::string& tensor_name(size_t index) const
    {
        static const std::string kEmpty;
        return index < metas_.size() ? metas_[index].name : kEmpty;
    }

    /// Get all tensor metadata
    const std::vector<TensorMeta>& tensors() const { return metas_; }

    /// Check if a tensor exists
    bool has_tensor(const std::string& name) const
    {
        for (const auto& m : metas_) {
            if (m.name == name) return true;
        }
        return false;
    }

    /// Find tensor index by name, returns -1 if not found
    int find_tensor(const std::string& name) const
    {
        for (size_t i = 0; i < metas_.size(); ++i) {
            if (metas_[i].name == name) return static_cast<int>(i);
        }
        return -1;
    }

    /// Get tensor metadata by name (returns nullptr if not found)
    const TensorMeta* get_tensor_meta(const std::string& name) const
    {
        for (const auto& m : metas_) {
            if (m.name == name) return &m;
        }
        return nullptr;
    }

    /// Read tensor data into caller-provided buffer
    /// Returns number of bytes read, or 0 on error
    size_t read_tensor(const std::string& name, void* buffer, size_t buffer_size) const
    {
        const auto* meta = get_tensor_meta(name);
        if (!meta) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        if (buffer_size < meta->size) {
            throw std::runtime_error("Buffer too small for tensor: " + name);
        }

        // Calculate absolute offset: 8-byte header + header_size + data_offset
        size_t abs_offset = 8 + header_size_ + meta->offset;

        stream_.seekg(static_cast<std::streampos>(abs_offset));
        if (!stream_.good()) {
            throw std::runtime_error("Failed to seek to tensor data: " + name);
        }

        stream_.read(static_cast<char*>(buffer), static_cast<std::streamsize>(meta->size));
        if (!stream_.good()) {
            throw std::runtime_error("Failed to read tensor data: " + name);
        }

        return meta->size;
    }

    /// Read tensor data and return as a vector (caller owns the memory)
    std::vector<uint8_t> read_tensor(const std::string& name) const
    {
        const auto* meta = get_tensor_meta(name);
        if (!meta) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        std::vector<uint8_t> data(meta->size);
        read_tensor(name, data.data(), meta->size);
        return data;
    }

    /// Get file size
    size_t file_size() const { return file_size_; }

    /// Parse dtype string to DataType
    static DataType ParseDtype(const std::string& dtype)
    {
        if (dtype == "F32" || dtype == "fp32" || dtype == "float32") return DataType::kFloat32;
        if (dtype == "F16" || dtype == "fp16" || dtype == "float16") return DataType::kFloat16;
        if (dtype == "BF16" || dtype == "bf16" || dtype == "bfloat16") return DataType::kBfloat16;
        if (dtype == "I64" || dtype == "i64" || dtype == "int64") return DataType::kInt64;
        if (dtype == "I32" || dtype == "i32" || dtype == "int32") return DataType::kInt32;
        if (dtype == "I16" || dtype == "i16" || dtype == "int16") return DataType::kInt16;
        if (dtype == "I8" || dtype == "i8" || dtype == "int8") return DataType::kInt8;
        if (dtype == "U64" || dtype == "u64" || dtype == "uint64") return DataType::kUint64;
        if (dtype == "U32" || dtype == "u32" || dtype == "uint32") return DataType::kUint32;
        if (dtype == "U16" || dtype == "u16" || dtype == "uint16") return DataType::kUint16;
        if (dtype == "U8" || dtype == "u8" || dtype == "uint8") return DataType::kUint8;
        if (dtype == "F64" || dtype == "fp64" || dtype == "float64") return DataType::kFloat64;
        if (dtype == "BOOL" || dtype == "bool") return DataType::kBool;
        // Default to float32 for unknown dtypes
        return DataType::kFloat32;
    }

private:
    /// Parse the safetensors header (8-byte size + JSON metadata)
    void ParseHeader()
    {
        // Get file size
        stream_.seekg(0, std::ios::end);
        file_size_ = static_cast<size_t>(stream_.tellg());
        stream_.seekg(0, std::ios::beg);

        // Read 8-byte header size (little-endian uint64)
        uint8_t size_bytes[8] = {0};
        stream_.read(reinterpret_cast<char*>(size_bytes), 8);
        header_size_ = 0;
        for (int i = 0; i < 8; ++i) {
            header_size_ |= static_cast<size_t>(size_bytes[i]) << (i * 8);
        }

        if (header_size_ == 0 || 8 + header_size_ > file_size_) {
            throw std::runtime_error("Invalid safetensors header size");
        }

        // Read header JSON
        std::vector<char> header_json(header_size_);
        stream_.read(header_json.data(), static_cast<std::streamsize>(header_size_));

        // Parse JSON to extract tensor metadata
        std::string json_str(header_json.begin(), header_json.end());
        ParseJsonHeader(json_str);
    }

    /// Parse safetensors JSON header
    /// Format: {"tensor_name": {"dtype": "F32", "shape": [1, 768], "data_offsets": [0, 3072]}}
    void ParseJsonHeader(const std::string& json)
    {
        size_t pos = 0;

        // Skip leading whitespace and opening brace
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                     json[pos] == '\n' || json[pos] == '\r')) {
            pos++;
        }
        if (pos >= json.size() || json[pos] != '{') {
            throw std::runtime_error("Invalid safetensors JSON: expected '{'");
        }
        pos++;  // Skip '{'

        while (pos < json.size()) {
            // Skip whitespace
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                         json[pos] == '\n' || json[pos] == '\r')) {
                pos++;
            }
            if (pos >= json.size()) break;

            // Check for closing brace
            if (json[pos] == '}') break;

            // Skip comma
            if (json[pos] == ',') {
                pos++;
                continue;
            }

            // Parse tensor name
            if (json[pos] != '"') {
                throw std::runtime_error("Invalid safetensors JSON: expected '\"' at pos " + std::to_string(pos));
            }
            pos++;  // Skip opening quote
            size_t name_start = pos;
            while (pos < json.size() && json[pos] != '"') {
                pos++;
            }
            if (pos >= json.size()) {
                throw std::runtime_error("Invalid safetensors JSON: unclosed string");
            }
            std::string tensor_name = json.substr(name_start, pos - name_start);
            pos++;  // Skip closing quote

            // Skip to colon
            while (pos < json.size() && json[pos] != ':') pos++;
            if (pos >= json.size()) break;
            pos++;  // Skip ':'

            // Skip whitespace
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

            // Skip __metadata__ entries
            if (tensor_name == "__metadata__") {
                // Find matching closing brace at same depth
                int depth = 1;
                pos++;
                while (pos < json.size() && depth > 0) {
                    if (json[pos] == '{') depth++;
                    else if (json[pos] == '}') depth--;
                    pos++;
                }
                continue;
            }

            // Parse tensor object { ... }
            if (json[pos] != '{') {
                throw std::runtime_error("Invalid safetensors JSON: expected '{' for tensor " + tensor_name);
            }
            pos++;  // Skip '{'

            TensorMeta meta;
            meta.name = tensor_name;
            meta.dtype = DataType::kFloat32;  // Default dtype - will be overwritten if found
            meta.offset = 0;
            meta.size = 0;

            // Parse dtype, shape, data_offsets within the tensor object
            while (pos < json.size()) {
                // Skip whitespace
                while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                             json[pos] == '\n' || json[pos] == '\r')) {
                    pos++;
                }
                if (pos >= json.size()) break;

                // Check for end of tensor object
                if (json[pos] == '}') {
                    pos++;  // Skip '}'
                    break;
                }

                // Skip comma
                if (json[pos] == ',') {
                    pos++;
                    continue;
                }

                // Parse key
                if (json[pos] != '"') {
                    pos++;
                    continue;
                }
                pos++;  // Skip opening quote
                size_t key_start = pos;
                while (pos < json.size() && json[pos] != '"') pos++;
                if (pos >= json.size()) break;
                std::string key = json.substr(key_start, pos - key_start);
                pos++;  // Skip closing quote

                // Skip to colon
                while (pos < json.size() && json[pos] != ':') pos++;
                if (pos >= json.size()) break;
                pos++;  // Skip ':'

                // Skip whitespace
                while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

                // Parse value based on key
                if (key == "dtype") {
                    // String value
                    if (json[pos] != '"') {
                        throw std::runtime_error("Invalid dtype: expected '\"'");
                    }
                    pos++;  // Skip opening quote
                    size_t dtype_start = pos;
                    while (pos < json.size() && json[pos] != '"') pos++;
                    std::string dtype_str = json.substr(dtype_start, pos - dtype_start);
                    meta.dtype = ParseDtype(dtype_str);
                    pos++;  // Skip closing quote
                }
                else if (key == "shape") {
                    // Array value
                    if (json[pos] != '[') {
                        throw std::runtime_error("Invalid shape: expected '['");
                    }
                    pos++;  // Skip '['
                    while (pos < json.size()) {
                        // Skip whitespace
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        if (pos >= json.size()) break;

                        // Check for end of array
                        if (json[pos] == ']') {
                            pos++;  // Skip ']'
                            break;
                        }

                        // Parse number
                        size_t num_start = pos;
                        while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-')) {
                            pos++;
                        }
                        std::string num_str = json.substr(num_start, pos - num_start);
                        if (!num_str.empty()) {
                            meta.shape.push_back(static_cast<size_t>(std::stoll(num_str)));
                        }

                        // Skip whitespace and comma
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        if (json[pos] == ',') {
                            pos++;
                        }
                    }
                }
                else if (key == "data_offsets") {
                    // Array with 2 values: [start, end]
                    if (json[pos] != '[') {
                        throw std::runtime_error("Invalid data_offsets: expected '['");
                    }
                    pos++;  // Skip '['
                    size_t start = 0, end = 0;
                    for (int i = 0; i < 2; ++i) {
                        // Skip whitespace
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        size_t num_start = pos;
                        while (pos < json.size() && std::isdigit(json[pos])) {
                            pos++;
                        }
                        if (i == 0) {
                            start = static_cast<size_t>(std::stoll(json.substr(num_start, pos - num_start)));
                        } else {
                            end = static_cast<size_t>(std::stoll(json.substr(num_start, pos - num_start)));
                        }
                        // Skip whitespace and comma
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        if (json[pos] == ',') pos++;
                    }
                    // Skip closing bracket
                    while (pos < json.size() && json[pos] != ']') pos++;
                    if (pos < json.size()) pos++;

                    meta.offset = start;
                    meta.size = end - start;
                }
                else {
                    // Unknown key, skip to comma or }
                    while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
                    if (pos < json.size() && json[pos] == ',') pos++;
                }
            }

            metas_.push_back(meta);
        }
    }

    mutable std::ifstream stream_;
    mutable size_t file_size_ = 0;
    mutable size_t header_size_ = 0;
    mutable std::vector<TensorMeta> metas_;
};

}  // namespace turbomind