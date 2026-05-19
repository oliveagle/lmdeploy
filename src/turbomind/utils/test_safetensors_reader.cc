// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/utils/safetensors_reader.h"

#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <cstring>

namespace turbomind {

void CreateTestSafetensorsFile(const std::string& path)
{
    // Create a minimal safetensors file for testing
    // Format: 8-byte header size + JSON header + binary data

    // JSON header with two tensors
    std::string json_header = R"({
        "tensor1": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        },
        "tensor2": {
            "dtype": "I32",
            "shape": [4],
            "data_offsets": [24, 40]
        }
    })";

    // Remove whitespace for compact header (optional, but saves space)
    // For simplicity, we'll keep the whitespace

    uint64_t header_size = json_header.size();

    // Prepare tensor data
    std::vector<float> tensor1_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int32_t> tensor2_data = {10, 20, 30, 40};

    // Write to file
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to create test file");
    }

    // Write header size (little-endian)
    out.write(reinterpret_cast<const char*>(&header_size), 8);

    // Write JSON header
    out.write(json_header.data(), json_header.size());

    // Write tensor data
    out.write(reinterpret_cast<const char*>(tensor1_data.data()), tensor1_data.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(tensor2_data.data()), tensor2_data.size() * sizeof(int32_t));

    out.close();
}

void TestSafetensorsReader()
{
    std::cout << "Testing SafetensorsReader..." << std::endl;

    // Create a test file
    std::string test_path = "/tmp/test_model.safetensors";
    CreateTestSafetensorsFile(test_path);

    // Test 1: Open file and parse header
    {
        SafetensorsReader reader(test_path);
        assert(reader.num_tensors() == 2);
        std::cout << "  [PASS] File opening and header parsing" << std::endl;
    }

    // Test 2: Tensor names
    {
        SafetensorsReader reader(test_path);
        assert(reader.tensor_name(0) == "tensor1");
        assert(reader.tensor_name(1) == "tensor2");
        std::cout << "  [PASS] Tensor name access" << std::endl;
    }

    // Test 3: has_tensor and find_tensor
    {
        SafetensorsReader reader(test_path);
        assert(reader.has_tensor("tensor1") == true);
        assert(reader.has_tensor("tensor2") == true);
        assert(reader.has_tensor("nonexistent") == false);
        assert(reader.find_tensor("tensor1") == 0);
        assert(reader.find_tensor("tensor2") == 1);
        assert(reader.find_tensor("nonexistent") == -1);
        std::cout << "  [PASS] has_tensor and find_tensor" << std::endl;
    }

    // Test 4: Tensor metadata
    {
        SafetensorsReader reader(test_path);
        const auto* meta1 = reader.get_tensor_meta("tensor1");
        assert(meta1 != nullptr);
        assert(meta1->name == "tensor1");
        assert(meta1->shape.size() == 2);
        assert(meta1->shape[0] == 2);
        assert(meta1->shape[1] == 3);
        assert(meta1->offset == 0);
        assert(meta1->size == 24);
        assert(meta1->dtype == DataType::kFloat32);

        const auto* meta2 = reader.get_tensor_meta("tensor2");
        assert(meta2 != nullptr);
        assert(meta2->name == "tensor2");
        assert(meta2->shape.size() == 1);
        assert(meta2->shape[0] == 4);
        assert(meta2->offset == 24);
        assert(meta2->size == 16);
        assert(meta2->dtype == DataType::kInt32);

        std::cout << "  [PASS] Tensor metadata" << std::endl;
    }

    // Test 5: Read tensor data
    {
        SafetensorsReader reader(test_path);

        // Read tensor1
        std::vector<uint8_t> data1 = reader.read_tensor("tensor1");
        assert(data1.size() == 24);
        std::vector<float> tensor1_values(6);
        std::memcpy(tensor1_values.data(), data1.data(), 24);
        assert(tensor1_values[0] == 1.0f);
        assert(tensor1_values[1] == 2.0f);
        assert(tensor1_values[2] == 3.0f);
        assert(tensor1_values[3] == 4.0f);
        assert(tensor1_values[4] == 5.0f);
        assert(tensor1_values[5] == 6.0f);

        // Read tensor2
        std::vector<uint8_t> data2 = reader.read_tensor("tensor2");
        assert(data2.size() == 16);
        std::vector<int32_t> tensor2_values(4);
        std::memcpy(tensor2_values.data(), data2.data(), 16);
        assert(tensor2_values[0] == 10);
        assert(tensor2_values[1] == 20);
        assert(tensor2_values[2] == 30);
        assert(tensor2_values[3] == 40);

        std::cout << "  [PASS] Tensor data reading" << std::endl;
    }

    // Test 6: Read tensor into buffer
    {
        SafetensorsReader reader(test_path);
        std::vector<float> buffer(6);
        size_t bytes_read = reader.read_tensor("tensor1", buffer.data(), buffer.size() * sizeof(float));
        assert(bytes_read == 24);
        assert(buffer[0] == 1.0f);
        assert(buffer[1] == 2.0f);
        assert(buffer[2] == 3.0f);
        assert(buffer[3] == 4.0f);
        assert(buffer[4] == 5.0f);
        assert(buffer[5] == 6.0f);
        std::cout << "  [PASS] Read tensor into buffer" << std::endl;
    }

    // Test 7: Error handling - nonexistent tensor
    {
        SafetensorsReader reader(test_path);
        bool caught = false;
        try {
            reader.read_tensor("nonexistent");
        }
        catch (const std::runtime_error&) {
            caught = true;
        }
        assert(caught);
        std::cout << "  [PASS] Error handling for nonexistent tensor" << std::endl;
    }

    // Test 8: Dtype parsing
    {
        // Test various dtype strings
        assert(SafetensorsReader::ParseDtype("F32") == DataType::kFloat32);
        assert(SafetensorsReader::ParseDtype("fp16") == DataType::kFloat16);
        assert(SafetensorsReader::ParseDtype("BF16") == DataType::kBfloat16);
        assert(SafetensorsReader::ParseDtype("I64") == DataType::kInt64);
        assert(SafetensorsReader::ParseDtype("U8") == DataType::kUint8);
        std::cout << "  [PASS] Dtype parsing" << std::endl;
    }

    // Test 9: File size
    {
        SafetensorsReader reader(test_path);
        assert(reader.file_size() > 0);
        std::cout << "  [PASS] File size reporting" << std::endl;
    }

    // Cleanup
    std::remove(test_path.c_str());

    std::cout << "All tests passed!" << std::endl;
}

}  // namespace turbomind

int main()
{
    turbomind::TestSafetensorsReader();
    return 0;
}
