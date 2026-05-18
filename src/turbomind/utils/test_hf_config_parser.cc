// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/utils/hf_config_parser.h"

#include <iostream>
#include <cassert>
#include <fstream>

namespace turbomind {

void TestHfConfigParser() {
    std::cout << "Testing HfConfigParser..." << std::endl;

    // Test 1: Simple object parsing
    {
        std::string json = R"({"hidden_size": 4096, "num_layers": 32})";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.is_object());
        assert(result.get("hidden_size").as_int() == 4096);
        assert(result.get("num_layers").as_int() == 32);
        std::cout << "  [PASS] Simple object parsing" << std::endl;
    }

    // Test 2: Nested objects (text_config pattern)
    {
        std::string json = R"({
            "model_type": "qwen2",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 40
            }
        })";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.get("model_type").as_string() == "qwen2");
        assert(result.get("text_config.hidden_size").as_int() == 5120);
        assert(result.get("text_config.num_hidden_layers").as_int() == 40);
        std::cout << "  [PASS] Nested objects (text_config)" << std::endl;
    }

    // Test 3: Quantization config parsing
    {
        std::string json = R"({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm"
            }
        })";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.get("quantization_config.quant_method").as_string() == "awq");
        assert(result.get("quantization_config.bits").as_int() == 4);
        assert(result.get("quantization_config.group_size").as_int() == 128);
        std::cout << "  [PASS] Quantization config parsing" << std::endl;
    }

    // Test 4: Arrays
    {
        std::string json = R"({"shape": [1, 768, 1024]})";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        auto arr = result.get("shape").as_array();
        assert(arr.size() == 3);
        assert(arr[0].as_int() == 1);
        assert(arr[1].as_int() == 768);
        assert(arr[2].as_int() == 1024);
        std::cout << "  [PASS] Array parsing" << std::endl;
    }

    // Test 5: Booleans and null
    {
        std::string json = R"({
            "use_cache": true,
            "use_linear_attn": false,
            "optional_value": null
        })";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.get("use_cache").as_bool() == true);
        assert(result.get("use_linear_attn").as_bool() == false);
        assert(result.get("optional_value").is_null());
        std::cout << "  [PASS] Booleans and null" << std::endl;
    }

    // Test 6: Floats and negative numbers
    {
        std::string json = R"({
            "rms_norm_eps": 1e-06,
            "rope_theta": -1.0,
            "temperature": 0.7
        })";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.get("rms_norm_eps").as_float() == 1e-06);
        assert(result.get("rope_theta").as_float() == -1.0);
        assert(result.get("temperature").as_float() == 0.7);
        std::cout << "  [PASS] Floats and negative numbers" << std::endl;
    }

    // Test 7: String values with escapes
    {
        std::string json = R"({"model_name": "Qwen/Qwen2-7B-Instruct"})";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.get("model_name").as_string() == "Qwen/Qwen2-7B-Instruct");
        std::cout << "  [PASS] String values" << std::endl;
    }

    // Test 8: has() method
    {
        std::string json = R"({"hidden_size": 4096})";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.has("hidden_size") == true);
        assert(result.has("num_layers") == false);
        std::cout << "  [PASS] has() method" << std::endl;
    }

    // Test 9: Complex nested structure (like Qwen3.5 MoE)
    {
        std::string json = R"({
            "model_type": "qwen2_moe",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 40,
                "num_local_experts": 64,
                "num_experts_per_tok": 8
            },
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4
            }
        })";
        auto result = HfConfigParser::Parse(json);
        assert(!result.is_null());
        assert(result.get("model_type").as_string() == "qwen2_moe");
        assert(result.get("text_config.num_local_experts").as_int() == 64);
        assert(result.get("quantization_config.quant_method").as_string() == "awq");
        std::cout << "  [PASS] Complex nested structure" << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
}

}  // namespace turbomind

int main() {
    turbomind::TestHfConfigParser();
    return 0;
}
