/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.cc

#include <cuda_runtime.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/models/llama/DFlashDraftWeight.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/core/logger.h"

namespace turbomind {

LlamaWeight::LlamaWeight(DataType           data_type,
                         const ModelParam&  model,
                         const EngineParam& engine_param,
                         const MoeParam&    moe_param):
    model_param_{model},
    engine_param_{engine_param},
    moe_param_{moe_param},
    hidden_units_(model.hidden_units),
    inter_size_(model.inter_size),
    vocab_size_(model.vocab_size),
    vocab_size_padded_(model.vocab_size),
    embedding_size_(model.embedding_size),
    num_layer_(model.layer_num),
    data_type_{data_type},
    weight_type_{model.weight_type},
    tp_size_(engine_param.attn_tp_size * engine_param.attn_cp_size),
    tp_rank_(engine_param.attn_tp_rank * engine_param.attn_cp_size + engine_param.attn_cp_rank)
{
    if (vocab_size_padded_ % tp_size_ != 0) {
        vocab_size_padded_ = (vocab_size_ + tp_size_ - 1) / tp_size_ * tp_size_;
        TM_LOG_WARN("pad vocab size from {} to {}", vocab_size_, vocab_size_padded_);
    }
    if (embedding_size_ % tp_size_ != 0) {
        embedding_size_ = (embedding_size_ + tp_size_ - 1) / tp_size_ * tp_size_;
        TM_LOG_WARN("pad embed size from {} to {}", embedding_size_, embedding_size_);
    }
    FT_CHECK(hidden_units_ % tp_size_ == 0);
    TM_CHECK_EQ(vocab_size_padded_ % tp_size_, 0);
    TM_CHECK_EQ(hidden_units_ % tp_size_, 0);

    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, false};

    initialize();
}

LlamaWeight::~LlamaWeight()
{
    release();
}

bool LlamaWeight::is_initialized() const
{
    return initialized_;
}

void LlamaWeight::initialize()
{
    core::ContextGuard guard = context();

    pre_decoder_embedding.emplace(embedding_size_, hidden_units_ / tp_size_, data_type_, false, data_type_, 1);
    post_decoder_embedding.emplace(hidden_units_, vocab_size_padded_ / tp_size_, data_type_, false, data_type_, 1);
    register_module("tok_embeddings", pre_decoder_embedding, tp_rank_);
    register_module("output", post_decoder_embedding, tp_rank_);

    /// Lower VRAM pressure on consumer grade GPUs
    /// TODO: Support token embeds on pinned host memory
    pre_decoder_embedding.weight  = empty_like(pre_decoder_embedding.weight, kCPU);
    post_decoder_embedding.weight = empty_like(post_decoder_embedding.weight, kCPU);

    decoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
        decoder_layer_weights.emplace_back(
            new LlamaDecoderLayerWeight(data_type_, i, model_param_, engine_param_, moe_param_));
        register_module("layers", *decoder_layer_weights.back(), i);
    }

    output_norm_weight = Tensor{{hidden_units_}, data_type_, kDEVICE};
    register_parameter("norm.weight", output_norm_weight);
    initialized_ = true;
}

void LlamaWeight::release()
{
    core::ContextGuard guard = context();

    pre_decoder_embedding  = {};
    post_decoder_embedding = {};
    output_norm_weight     = {};

    for (auto& p : decoder_layer_weights) {
        delete p;
    }

    decoder_layer_weights.clear();
    pinned_weights_.clear();

    // Wait for deallocations
    core::Context::stream().Sync();

    // release memory back to os
    core::Context::device_alloc()->trim(0);
    initialized_ = false;
}

void LlamaWeight::to_device(const core::Device& device)
{
    TM_CHECK(device.type == kCPU || device.type == kDEVICE);
    core::ContextGuard guard{stream_, alloca_, Allocator{kCPUpinned}};

    auto tensor_ptr_map = get_parameters();
    for (auto& [name, tensor_ptr] : tensor_ptr_map) {
        if (device.type == kCPU) {
            if (pinned_weights_.find(name) == pinned_weights_.end()) {
                pinned_weights_[name] = empty_like(*tensor_ptr, kCPUpinned);
                Copy(*tensor_ptr, pinned_weights_[name]);
            }
            *tensor_ptr = {};
        }
        else {
            TM_CHECK(pinned_weights_.find(name) != pinned_weights_.end());
            *tensor_ptr = empty_like(pinned_weights_[name], kDEVICE);
            Copy(pinned_weights_[name], *tensor_ptr);
        }
    }
    core::Context::stream().Sync();
    if (device.type == kCPU) {
        core::Context::device_alloc()->trim(0);
    }
}

core::ContextGuard LlamaWeight::context() const
{
    return core::ContextGuard{stream_, alloca_};
}

void LlamaWeight::prepare(const cudaDeviceProp& prop)
{
    core::ContextGuard guard = context();

    // Wait for the weights to be filled externally
    check_cuda_error(cudaDeviceSynchronize());

    auto stream = core::Context::stream().handle();

    for (auto& layer : decoder_layer_weights) {
        layer->prepare(prop, stream);
    }

    auto to_device = [](Tensor& x) {
        auto tmp = std::exchange(x, empty_like(x, kDEVICE));
        Copy(tmp, x);
        return tmp;
    };

    // Keep the host tensor until stream synchronization
    auto tmp_token_embeds = to_device(pre_decoder_embedding.weight);
    auto tmp_lm_head      = to_device(post_decoder_embedding.weight);

    post_decoder_embedding.prepare();

    // Block until processing is done
    check_cuda_error(cudaStreamSynchronize(stream));
}

void LlamaWeight::LoadDFlashDraftWeight(const std::string& ckpt_path)
{
    /*
     * 加载 DFlash draft model 权重
     *
     * DFlash draft model 与 target model 共享以下权重:
     * - embed_tokens: token embedding
     * - lm_head: output projection
     *
     * DFlash 自己的权重从 checkpoint 加载:
     * - 8 层的 qkv_proj, o_proj, gate_up_proj, down_proj
     * - 8 层的 input_layernorm, post_attention_layernorm
     */

    TM_LOG_INFO("Loading DFlash draft model weights from: %s", ckpt_path.c_str());

    core::ContextGuard guard = context();

    // 创建 DFlash draft weight 结构
    dflash_draft_weight_ = std::make_unique<DFlashDraftWeight>();

    // 共享 embed_tokens 和 lm_head
    dflash_draft_weight_->embed_tokens = pre_decoder_embedding.weight;
    dflash_draft_weight_->lm_head = post_decoder_embedding.weight;

    // 设置配置 (从 ModelParam 获取)
    dflash_draft_weight_->num_layers = 8;
    dflash_draft_weight_->hidden_size = hidden_units_;
    dflash_draft_weight_->intermediate_size = inter_size_[0];  // 使用第一层的 inter_size
    dflash_draft_weight_->num_attention_heads = model_param_.head_num;
    dflash_draft_weight_->num_key_value_heads = model_param_.kv_head_num;
    dflash_draft_weight_->head_dim = model_param_.head_dim;
    dflash_draft_weight_->rms_norm_eps = model_param_.norm_eps;

    // DFlash 权重由 Python 通过 DLPack 加载, 这里只做初始化
    // Python 侧通过 SetDFlashWeights() 传入各层权重

    TM_LOG_INFO("DFlash draft model weights loaded:");
    TM_LOG_INFO("  num_layers: %d", dflash_draft_weight_->num_layers);
    TM_LOG_INFO("  hidden_size: %d", dflash_draft_weight_->hidden_size);
    TM_LOG_INFO("  intermediate_size: %d", dflash_draft_weight_->intermediate_size);
    TM_LOG_INFO("  num_attention_heads: %d", dflash_draft_weight_->num_attention_heads);
    TM_LOG_INFO("  num_key_value_heads: %d", dflash_draft_weight_->num_key_value_heads);
    TM_LOG_INFO("  head_dim: %d", dflash_draft_weight_->head_dim);
    TM_LOG_INFO("  rms_norm_eps: %f", dflash_draft_weight_->rms_norm_eps);
    TM_LOG_INFO("  embed_tokens: [shared from target model]");
    TM_LOG_INFO("  lm_head: [shared from target model]");
}

}  // namespace turbomind
