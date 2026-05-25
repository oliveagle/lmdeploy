// Comprehensive benchmark to identify the safetensors loading bottleneck
// Measures mmap, path mapping, module traversal, and I/O overhead

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <random>

// Simple JSON parser for safetensors header
struct TensorMeta {
    std::string name;
    std::vector<size_t> shape;
    size_t offset;
    size_t size;
};

class SafetensorsReader {
public:
    explicit SafetensorsReader(const char* file_path)
        : fd_(-1), mapped_data_(nullptr), file_size_(0), header_size_(0)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        fd_ = open(file_path, O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open file");
        }

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close(fd_);
            throw std::runtime_error("Cannot stat file");
        }
        file_size_ = static_cast<size_t>(st.st_size);

        auto t1 = std::chrono::high_resolution_clock::now();

        // mmap entire file
        mapped_data_ = static_cast<const uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapped_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Cannot mmap file");
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        ParseHeader();

        auto t3 = std::chrono::high_resolution_clock::now();

        auto open_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        auto mmap_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
        auto parse_ms = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;

        printf("[PHASE] open: %.2fms, mmap: %.2fms, parse: %.2fms\n",
               open_ms, mmap_ms, parse_ms);
    }

    ~SafetensorsReader() {
        if (mapped_data_ != MAP_FAILED && mapped_data_ != nullptr) {
            munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    size_t num_tensors() const { return metas_.size(); }
    size_t file_size() const { return file_size_; }
    const std::vector<TensorMeta>& tensors() const { return metas_; }

    const TensorMeta* get_tensor_meta(const std::string& name) const {
        auto it = meta_map_.find(name);
        return it != meta_map_.end() ? &it->second : nullptr;
    }

    const uint8_t* get_tensor_data(const std::string& name) const {
        const auto* meta = get_tensor_meta(name);
        if (!meta) return nullptr;
        size_t abs_offset = 8 + header_size_ + meta->offset;
        if (abs_offset + meta->size > file_size_) return nullptr;
        return mapped_data_ + abs_offset;
    }

private:
    void ParseHeader() {
        uint64_t header_size_le = 0;
        std::memcpy(&header_size_le, mapped_data_, 8);
        header_size_ = 0;
        for (int i = 0; i < 8; ++i) {
            header_size_ |= static_cast<size_t>(
                (reinterpret_cast<const uint8_t*>(&header_size_le))[i]) << (i * 8);
        }

        std::string json_str(reinterpret_cast<const char*>(mapped_data_ + 8), header_size_);
        ParseJsonHeader(json_str);
    }

    void ParseJsonHeader(const std::string& json) {
        size_t pos = 0;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                     json[pos] == '\n' || json[pos] == '\r')) {
            pos++;
        }
        if (pos >= json.size() || json[pos] != '{') return;
        pos++;

        while (pos < json.size()) {
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                         json[pos] == '\n' || json[pos] == '\r')) pos++;
            if (pos >= json.size()) break;
            if (json[pos] == '}') break;
            if (json[pos] == ',') { pos++; continue; }

            if (json[pos] != '"') continue;
            pos++;
            size_t name_start = pos;
            while (pos < json.size() && json[pos] != '"') pos++;
            if (pos >= json.size()) break;
            std::string tensor_name = json.substr(name_start, pos - name_start);
            pos++;

            while (pos < json.size() && json[pos] != ':') pos++;
            if (pos >= json.size()) break;
            pos++;

            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

            if (tensor_name == "__metadata__") {
                int depth = 1;
                pos++;
                while (pos < json.size() && depth > 0) {
                    if (json[pos] == '{') depth++;
                    else if (json[pos] == '}') depth--;
                    pos++;
                }
                continue;
            }

            if (json[pos] != '{') continue;
            pos++;

            TensorMeta meta;
            meta.name = tensor_name;
            meta.offset = 0;
            meta.size = 0;

            while (pos < json.size()) {
                while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' &&
                                             json[pos] == '\n' || json[pos] == '\r')) pos++;
                if (pos >= json.size()) break;
                if (json[pos] == '}') { pos++; break; }
                if (json[pos] == ',') { pos++; continue; }

                if (json[pos] != '"') { pos++; continue; }
                pos++;
                size_t key_start = pos;
                while (pos < json.size() && json[pos] != '"') pos++;
                if (pos >= json.size()) break;
                std::string key = json.substr(key_start, pos - key_start);
                pos++;

                while (pos < json.size() && json[pos] != ':') pos++;
                if (pos >= json.size()) break;
                pos++;

                while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

                if (key == "shape") {
                    if (json[pos] != '[') continue;
                    pos++;
                    while (pos < json.size()) {
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        if (pos >= json.size()) break;
                        if (json[pos] == ']') { pos++; break; }
                        size_t num_start = pos;
                        while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-')) pos++;
                        std::string num_str = json.substr(num_start, pos - num_start);
                        if (!num_str.empty()) {
                            meta.shape.push_back(static_cast<size_t>(std::stoll(num_str)));
                        }
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        if (json[pos] == ',') pos++;
                    }
                } else if (key == "data_offsets") {
                    if (json[pos] != '[') continue;
                    pos++;
                    size_t start = 0, end = 0;
                    for (int i = 0; i < 2; ++i) {
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        size_t num_start = pos;
                        while (pos < json.size() && std::isdigit(json[pos])) pos++;
                        if (i == 0) start = static_cast<size_t>(std::stoll(json.substr(num_start, pos - num_start)));
                        else end = static_cast<size_t>(std::stoll(json.substr(num_start, pos - num_start)));
                        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
                        if (json[pos] == ',') pos++;
                    }
                    while (pos < json.size() && json[pos] != ']') pos++;
                    if (pos < json.size()) pos++;
                    meta.offset = start;
                    meta.size = end - start;
                } else {
                    while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
                    if (pos < json.size() && json[pos] == ',') pos++;
                }
            }

            metas_.push_back(meta);
            meta_map_.insert({metas_.back().name, metas_.back()});
        }
    }

    int fd_;
    const uint8_t* mapped_data_;
    size_t file_size_;
    size_t header_size_;
    std::vector<TensorMeta> metas_;
    std::unordered_map<std::string, TensorMeta> meta_map_;
};

// Simulate the path mapping from MapHuggingFaceWeightToTurboMind
static std::string SimulatePathMapping(const std::string& hf_name) {
    std::string result = hf_name;

    if (result.find("model.") == 0) {
        result = result.substr(6);
    }
    if (result.find("language_model.") == 0) {
        result = result.substr(15);
    }

    size_t pos;
    while ((pos = result.find(".self_attn.in_proj.qkv.")) != std::string::npos) {
        result.replace(pos, 21, ".linear_attn.in_proj_qkv.");
    }
    while ((pos = result.find(".self_attn.in_proj.weight")) != std::string::npos) {
        result.replace(pos, 22, ".linear_attn.in_proj_all.weight");
    }
    while ((pos = result.find(".self_attn.")) != std::string::npos) {
        result.replace(pos, 11, ".attention.");
    }
    while ((pos = result.find(".mlp.experts.")) != std::string::npos) {
        result.replace(pos, 13, ".moe_ffn.experts.");
    }
    while ((pos = result.find(".mlp.")) != std::string::npos) {
        if (result.find(".moe_ffn.", pos - 5) != pos - 5) {
            result.replace(pos, 5, ".feed_forward.");
        } else {
            break;
        }
    }
    while ((pos = result.find(".input_layernorm")) != std::string::npos) {
        result.replace(pos, 16, ".attention_norm");
    }
    while ((pos = result.find(".post_attention_layernorm")) != std::string::npos) {
        result.replace(pos, 24, ".ffn_norm");
    }
    while ((pos = result.find(".q_proj.")) != std::string::npos) {
        result.replace(pos, 8, ".q_proj.");
    }
    while ((pos = result.find(".k_proj.")) != std::string::npos) {
        result.replace(pos, 8, ".k_proj.");
    }
    while ((pos = result.find(".v_proj.")) != std::string::npos) {
        result.replace(pos, 8, ".v_proj.");
    }
    while ((pos = result.find(".o_proj.")) != std::string::npos) {
        result.replace(pos, 8, ".wo.");
    }
    while ((pos = result.find(".gate_proj.")) != std::string::npos) {
        result.replace(pos, 11, ".w1.");
    }
    while ((pos = result.find(".up_proj.")) != std::string::npos) {
        result.replace(pos, 9, ".w3.");
    }
    while ((pos = result.find(".down_proj.")) != std::string::npos) {
        result.replace(pos, 11, ".w2.");
    }
    while ((pos = result.find(".qweight")) != std::string::npos) {
        result.replace(pos, 8, ".weight");
    }
    while ((pos = result.find(".qzeros")) != std::string::npos) {
        result.replace(pos, 7, ".zeros");
    }
    while ((pos = result.find(".weight_scale")) != std::string::npos) {
        result.replace(pos, 13, ".scales");
    }

    return result;
}

// Simulate module traversal (simplified - O(n) lookup)
static bool SimulateModuleTraversal(const std::string& path) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t dot_pos = path.find('.', start);
        if (dot_pos == std::string::npos) {
            parts.push_back(path.substr(start));
            break;
        }
        parts.push_back(path.substr(start, dot_pos - start));
        start = dot_pos + 1;
    }

    // Simulate O(n) traversal - in reality this is O(1) for LinearWeight params
    // but the module hierarchy traversal is O(1) per child()
    // The bottleneck might be in the ModuleList child() which iterates items_
    return !parts.empty();
}

// Measure fflush overhead
static void MeasureFflushOverhead(int iterations) {
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        fprintf(stderr, "[flush test %d]\n", i);
        fflush(stderr);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("[fflush benchmark] %d flushes took %ldms (%.2fms per flush)\n",
           iterations, ms, (double)ms / iterations);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <safetensors_file>\n", argv[0]);
        return 1;
    }

    const char* file_path = argv[1];

    // First, measure fflush overhead
    printf("=== Measuring fflush overhead ===\n");
    MeasureFflushOverhead(1000);
    MeasureFflushOverhead(5000);

    printf("\n=== Benchmarking safetensors: %s ===\n", file_path);

    try {
        auto total_start = std::chrono::high_resolution_clock::now();

        SafetensorsReader reader(file_path);

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

        printf("File size: %.2f GB\n", reader.file_size() / 1024.0 / 1024.0 / 1024.0);
        printf("Num tensors: %zu\n", reader.num_tensors());
        printf("Total load time: %ldms\n\n", total_ms);

        size_t total_size = 0;
        for (size_t i = 0; i < reader.num_tensors(); ++i) {
            total_size += reader.tensors()[i].size;
        }
        printf("Total tensor data: %.2f GB\n\n", total_size / 1024.0 / 1024.0 / 1024.0);

        // Benchmark 1: Path mapping
        printf("=== Benchmark 1: Path mapping ===\n");
        auto b1_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < reader.num_tensors(); ++i) {
            const auto& meta = reader.tensors()[i];
            std::string mapped = SimulatePathMapping(meta.name);
        }

        auto b1_end = std::chrono::high_resolution_clock::now();
        auto b1_ms = std::chrono::duration_cast<std::chrono::microseconds>(b1_end - b1_start).count() / 1000.0;
        printf("Path mapping: %.2fms (%.3fms per tensor)\n\n",
               b1_ms, b1_ms / reader.num_tensors());

        // Benchmark 2: Module traversal simulation
        printf("=== Benchmark 2: Module traversal ===\n");
        auto b2_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < reader.num_tensors(); ++i) {
            const auto& meta = reader.tensors()[i];
            std::string mapped = SimulatePathMapping(meta.name);
            SimulateModuleTraversal(mapped);
        }

        auto b2_end = std::chrono::high_resolution_clock::now();
        auto b2_ms = std::chrono::duration_cast<std::chrono::microseconds>(b2_end - b2_start).count() / 1000.0;
        printf("Module traversal: %.2fms (%.3fms per tensor)\n\n",
               b2_ms, b2_ms / reader.num_tensors());

        // Benchmark 3: get_tensor_data
        printf("=== Benchmark 3: get_tensor_data (mmap access) ===\n");
        auto b3_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < reader.num_tensors(); ++i) {
            const auto& meta = reader.tensors()[i];
            const uint8_t* data = reader.get_tensor_data(meta.name);
        }

        auto b3_end = std::chrono::high_resolution_clock::now();
        auto b3_ms = std::chrono::duration_cast<std::chrono::microseconds>(b3_end - b3_start).count() / 1000.0;
        printf("get_tensor_data: %.2fms (%.3fms per tensor)\n\n",
               b3_ms, b3_ms / reader.num_tensors());

        // Benchmark 4: Simulate FULL loading cycle per tensor (like LoadWeightsFromSafetensors)
        printf("=== Benchmark 4: Full loading cycle (simulating LoadWeightsFromSafetensors) ===\n");
        auto b4_start = std::chrono::high_resolution_clock::now();

        std::vector<size_t> shape_vec;
        shape_vec.reserve(4);

        for (size_t i = 0; i < reader.num_tensors(); ++i) {
            const auto& meta = reader.tensors()[i];

            // Step 1: Map path
            std::string tm_path = SimulatePathMapping(meta.name);

            // Step 2: Parse path
            std::vector<std::string> parts;
            size_t start = 0;
            while (true) {
                size_t dot_pos = tm_path.find('.', start);
                if (dot_pos == std::string::npos) {
                    parts.push_back(tm_path.substr(start));
                    break;
                }
                parts.push_back(tm_path.substr(start, dot_pos - start));
                start = dot_pos + 1;
            }

            // Step 3: Simulate module traversal
            SimulateModuleTraversal(tm_path);

            // Step 4: Build shape
            shape_vec.clear();
            for (auto s : meta.shape) {
                shape_vec.push_back(static_cast<size_t>(s));
            }

            // Step 5: Get data
            const uint8_t* data = reader.get_tensor_data(meta.name);
        }

        auto b4_end = std::chrono::high_resolution_clock::now();
        auto b4_ms = std::chrono::duration_cast<std::chrono::microseconds>(b4_end - b4_start).count() / 1000.0;
        printf("Full cycle: %.2fms (%.3fms per tensor)\n\n\n",
               b4_ms, b4_ms / reader.num_tensors());

        // Estimate: If we had debug printing for each tensor (like LoadWeightsFromSafetensors)
        printf("=== Estimate: Impact of debug printing ===\n");
        int flushes_per_tensor = 10;  // Estimate from LoadWeightsFromSafetensors code
        int tensors = reader.num_tensors();
        int total_flushes = tensors * flushes_per_tensor;
        double estimated_flush_ms = total_flushes * 0.02;  // ~0.02ms per flush
        printf("If %.2f flushes per tensor (%.2f total): ~%.2fms\n",
               (double)flushes_per_tensor, (double)total_flushes, estimated_flush_ms);

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}