// Copyright (c) OpenMMLab. All rights reserved.
// Mmap-based safetensors file reader for TurboMind
// Uses mmap for O(1) random access instead of sequential seek+read

#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "src/turbomind/core/data_type.h"

namespace turbomind {

/// Mmap-based safetensors file reader
/// Uses memory mapping for fast random access to tensor data
class SafetensorsReaderMmap {
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
    explicit SafetensorsReaderMmap(const char* file_path)
        : fd_(-1), mapped_data_(nullptr), file_size_(0), header_size_(0)
    {
        // Open file
        fd_ = open(file_path, O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error(std::string("Cannot open safetensors file: ") + file_path);
        }

        // Get file size
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close(fd_);
            throw std::runtime_error(std::string("Cannot stat safetensors file: ") + file_path);
        }
        file_size_ = static_cast<size_t>(st.st_size);

        // Map entire file into memory
        mapped_data_ = static_cast<const uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapped_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error(std::string("Cannot mmap safetensors file: ") + file_path);
        }

        // Parse header
        ParseHeader();
    }

    explicit SafetensorsReaderMmap(const std::string& file_path) : SafetensorsReaderMmap(file_path.c_str()) {}

    ~SafetensorsReaderMmap()
    {
        if (mapped_data_ != MAP_FAILED && mapped_data_ != nullptr) {
            munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    // Non-copyable, movable
    SafetensorsReaderMmap(const SafetensorsReaderMmap&) = delete;
    SafetensorsReaderMmap& operator=(const SafetensorsReaderMmap&) = delete;
    SafetensorsReaderMmap(SafetensorsReaderMmap&& other) noexcept
        : fd_(other.fd_), mapped_data_(other.mapped_data_), file_size_(other.file_size_),
          header_size_(other.header_size_), metas_(std::move(other.metas_))
    {
        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
    }

    SafetensorsReaderMmap& operator=(SafetensorsReaderMmap&& other) noexcept
    {
        if (this != &other) {
            if (mapped_data_ != MAP_FAILED && mapped_data_ != nullptr) {
                munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
            }
            if (fd_ >= 0) {
                close(fd_);
            }
            fd_ = other.fd_;
            mapped_data_ = other.mapped_data_;
            file_size_ = other.file_size_;
            header_size_ = other.header_size_;
            metas_ = std::move(other.metas_);
            other.fd_ = -1;
            other.mapped_data_ = nullptr;
            other.file_size_ = 0;
        }
        return *this;
    }

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

    /// Read tensor data directly from mmap (zero-copy!)
    /// Returns pointer to mapped memory, valid until reader is destroyed
    const uint8_t* get_tensor_data(const std::string& name) const
    {
        const auto* meta = get_tensor_meta(name);
        if (!meta) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        // Calculate absolute offset: 8-byte header + header_size + data_offset
        size_t abs_offset = 8 + header_size_ + meta->offset;

        if (abs_offset + meta->size > file_size_) {
            throw std::runtime_error("Tensor data out of bounds: " + name);
        }

        return mapped_data_ + abs_offset;
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

        const uint8_t* data = get_tensor_data(name);
        std::memcpy(buffer, data, meta->size);
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
        // Read 8-byte header size (little-endian uint64_t)
        uint64_t header_size_le = 0;
        std::memcpy(&header_size_le, mapped_data_, 8);
        header_size_ = 0;
        for (int i = 0; i < 8; ++i) {
            header_size_ |= static_cast<size_t>((reinterpret_cast<const uint8_t*>(&header_size_le))[i]) << (i * 8);
        }

        if (header_size_ == 0 || 8 + header_size_ > file_size_) {
            throw std::runtime_error("Invalid safetensors header size");
        }

        // Read header JSON directly from mmap
        std::string json_str(reinterpret_cast<const char*>(mapped_data_ + 8), header_size_);
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

    int fd_;                              // File descriptor
    const uint8_t* mapped_data_;          // Mapped file data
    size_t file_size_;                    // Total file size
    size_t header_size_;                  // JSON header size
    std::vector<TensorMeta> metas_;       // Tensor metadata
};

}  // namespace turbomind