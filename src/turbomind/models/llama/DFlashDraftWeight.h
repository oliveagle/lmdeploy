
#pragma once

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/models/llama/llama_params.h"
#include <vector>

namespace turbomind {

/**
 * DFlash draft model 权重结构
 *
 * 权重共享:
 * - embed_tokens: 从 target model 共享
 * - lm_head: 从 target model 共享
 *
 * DFlash 自己的权重:
 * - 8 层 decoder 的 qkv_proj, o_proj, gate_up_proj, down_proj
 * - 8 层的 input_layernorm, post_attention_layernorm
 */
struct DFlashDraftWeight {
    // 共享的权重 (从 target model)
    Tensor embed_tokens;   // [vocab_size, hidden_size]
    Tensor lm_head;         // [hidden_size, vocab_size]

    // CPU 权重 (从 Python 通过 DLPack 传入)
    std::vector<Tensor> attn_qkv_weight;  // [8][hidden * 3 * hidden]
    std::vector<Tensor> attn_qkv_bias;    // [8][3 * hidden]            (optional)
    std::vector<Tensor> attn_o_weight;    // [8][hidden * hidden]
    std::vector<Tensor> attn_o_bias;      // [8][hidden]                (optional)
    std::vector<Tensor> input_layernorm;  // [8][hidden]
    std::vector<Tensor> gate_up_proj;     // [8][intermediate*2 * hidden]
    std::vector<Tensor> down_proj;        // [8][hidden * intermediate]
    std::vector<Tensor> post_layernorm;   // [8][hidden]

    // GPU 权重 (由 ToDevice 填充)
    std::vector<Tensor> d_attn_qkv_weight;
    std::vector<Tensor> d_attn_qkv_bias;
    std::vector<Tensor> d_attn_o_weight;
    std::vector<Tensor> d_attn_o_bias;
    std::vector<Tensor> d_input_layernorm;
    std::vector<Tensor> d_gate_up_proj;
    std::vector<Tensor> d_down_proj;
    std::vector<Tensor> d_post_layernorm;

    // 配置
    int num_layers = 8;
    int hidden_size = 5120;
    int intermediate_size = 13824;
    int num_attention_heads = 40;
    int num_key_value_heads = 40;
    int head_dim = 128;
    float rms_norm_eps = 1e-5;

    DFlashDraftWeight();

    ~DFlashDraftWeight();

    void ToDevice(core::Allocator* allocator, const std::string& prefix);
};

}  // namespace turbomind
