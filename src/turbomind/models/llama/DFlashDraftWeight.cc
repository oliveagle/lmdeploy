/**
 * @file DFlashDraftWeight.cc
 *
 * DFlash Speculative Decoder - Weight Management
 *
 * @license Apache-2.0 WITH LLVM-exception
 *
 * Derived from and inspired by:
 * - lucebox-hub/dflash (https://github.com/lucebox-hub/dflash)
 *   Original authors: lucebox-hub contributors
 *   License: Apache-2.0
 *
 * @see DFlashDraftModel.h for full attribution
 */

#include "src/turbomind/models/llama/DFlashDraftWeight.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/logger.h"

#include <sstream>

namespace turbomind {

// ──────────────────────────────────────────
// Copy CPU tensor to GPU
// ──────────────────────────────────────────
static Tensor _ToDevice(const Tensor& src)
{
    if (!src) {
        return Tensor{};
    }
    Tensor dst(src.layout(), src.dtype(), kDEVICE);
    cudaMemcpy(dst.raw_data(), src.raw_data(), src.byte_size(), cudaMemcpyHostToDevice);
    return dst;
}

DFlashDraftWeight::DFlashDraftWeight()
{
    int n = num_layers;
    attn_qkv_weight.resize(n);
    attn_qkv_bias.resize(n);
    attn_o_weight.resize(n);
    attn_o_bias.resize(n);
    input_layernorm.resize(n);
    gate_up_proj.resize(n);
    down_proj.resize(n);
    post_layernorm.resize(n);

    d_attn_qkv_weight.resize(n);
    d_attn_qkv_bias.resize(n);
    d_attn_o_weight.resize(n);
    d_attn_o_bias.resize(n);
    d_input_layernorm.resize(n);
    d_gate_up_proj.resize(n);
    d_down_proj.resize(n);
    d_post_layernorm.resize(n);

    // Quantization scale vectors (STORY-008)
    d_qkv_scale.resize(n);
    d_o_scale.resize(n);
    d_gate_up_scale.resize(n);
    d_down_scale.resize(n);
}

DFlashDraftWeight::~DFlashDraftWeight() = default;

// ──────────────────────────────────────────
// Helper: parse checkpoint file into weight tensors
// ──────────────────────────────────────────
static bool _LoadNpy(const std::string& path, Tensor& out)
{
    (void)path;
    (void)out;
    return false;
}

void DFlashDraftWeight::ToDevice(core::Allocator* allocator,
                                 const std::string& prefix)
{
    (void)allocator;
    (void)prefix;

    auto move = [&](Tensor& dst, const Tensor& src) {
        if (src) {
            dst = _ToDevice(src);
        }
    };

    for (int i = 0; i < num_layers; ++i) {
        move(d_attn_qkv_weight[i],  attn_qkv_weight[i]);
        move(d_attn_qkv_bias[i],    attn_qkv_bias[i]);
        move(d_attn_o_weight[i],    attn_o_weight[i]);
        move(d_attn_o_bias[i],      attn_o_bias[i]);
        move(d_input_layernorm[i],  input_layernorm[i]);
        move(d_gate_up_proj[i],     gate_up_proj[i]);
        move(d_down_proj[i],        down_proj[i]);
        move(d_post_layernorm[i],   post_layernorm[i]);
    }

    TM_LOG_INFO("DFlashDraftWeight::ToDevice done, {} layers", num_layers);
}

}  // namespace turbomind
