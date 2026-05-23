// Minimal test to check if SafetensorsReaderMmap hangs
// Without using turbomind headers - just the reader logic

#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    std::cerr << "[TEST] Starting SafetensorsReaderMmap test..." << std::endl;

    const char* path = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ/model-00001-of-00008.safetensors";

    std::cerr << "[TEST] Opening file: " << path << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "[TEST] Cannot open file" << std::endl;
        return 1;
    }

    std::cerr << "[TEST] File opened, getting size..." << std::endl;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "[TEST] Cannot stat file" << std::endl;
        return 1;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    std::cerr << "[TEST] File size: " << (file_size / (1024 * 1024)) << " MB, mmap-ing..." << std::endl;

    void* mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        std::cerr << "[TEST] Cannot mmap file" << std::endl;
        close(fd);
        return 1;
    }

    std::cerr << "[TEST] mmap successful, reading header..." << std::endl;

    // Read header size
    uint64_t header_size_le = 0;
    std::memcpy(&header_size_le, mapped_data, 8);
    size_t header_size = 0;
    for (int i = 0; i < 8; ++i) {
        header_size |= static_cast<size_t>((reinterpret_cast<const uint8_t*>(&header_size_le))[i]) << (i * 8);
    }

    std::cerr << "[TEST] Header size: " << header_size << " bytes" << std::endl;

    if (header_size == 0 || 8 + header_size > file_size) {
        std::cerr << "[TEST] Invalid header size" << std::endl;
        return 1;
    }

    // Parse JSON header to count tensors
    std::string json_str(reinterpret_cast<const char*>(static_cast<uint8_t*>(mapped_data) + 8), header_size);

    std::cerr << "[TEST] JSON header length: " << json_str.size() << " characters" << std::endl;

    // Count tensor entries (count "dtype" keys)
    int tensor_count = 0;
    std::string search_key = "\"dtype\"";
    size_t pos = 0;
    while ((pos = json_str.find(search_key, pos)) != std::string::npos) {
        tensor_count++;
        pos += search_key.size();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cerr << "[TEST] Found ~" << tensor_count << " tensors in " << ms.count() << "ms" << std::endl;
    std::cerr << "[TEST] Header text (first 200 chars): " << json_str.substr(0, 200) << "..." << std::endl;

    std::cerr << "[TEST] Unmapping..." << std::endl;
    munmap(mapped_data, file_size);
    close(fd);

    std::cerr << "[TEST] SUCCESS" << std::endl;
    return 0;
}