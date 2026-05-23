// Minimal test to check if SafetensorsReaderMmap hangs

#include <iostream>
#include <chrono>
#include "src/turbomind/utils/safetensors_reader_mmap.h"

int main() {
    std::cerr << "[TEST] Starting SafetensorsReaderMmap test..." << std::endl;

    const char* path = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ/model-00001-of-00008.safetensors";

    std::cerr << "[TEST] About to create SafetensorsReaderMmap..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    try {
        turbomind::SafetensorsReaderMmap reader(path);
        auto end = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cerr << "[TEST] SafetensorsReaderMmap created in " << ms.count() << "ms" << std::endl;
        std::cerr << "[TEST] num_tensors: " << reader.num_tensors() << std::endl;

        if (reader.num_tensors() > 0) {
            std::cerr << "[TEST] First tensor: " << reader.tensor_name(0) << std::endl;
        }

        std::cerr << "[TEST] SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[TEST] Exception: " << e.what() << std::endl;
        return 1;
    }
}