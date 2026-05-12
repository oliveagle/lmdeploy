

#include <functional>
#include <numeric>
#include <optional>

#include <cuda_runtime.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
// Temporarily disable DFlash to fix build
// #include "src/turbomind/models/llama/DFlashDraftModel.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/core/logger.h"

#include "src/turbomind/engine/request.h"

// #include "dbg.h"

namespace turbomind {

void UnifiedDecoder::Run(BatchOp op, int phase, TensorMap& env)
{
    attn_layer_->Run(op, phase, env);
    if (linear_attn_layer_) {
        linear_attn_layer_->Run(op, phase, env);
    }
}

UnifiedDecoder::UnifiedDecoder(const ModelParam&     model,
                               const EngineParam&    engine,
                               const AttentionParam& attn,
                               const MoeParam&       moe,
                               const Context&        ctx,
                               int                   phases):
    layer_num_(model.layer_num),
    hidden_units_(model.hidden_units),
    attn_tp_size_(engine.attn_tp_size),
    attn_dp_size_(engine.attn_dp_size),
    attn_dp_rank_(engine.attn_dp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    attn_tp_group_(ctx.comm.d_tp_group),
    rmsnorm_eps_(model.norm_eps),
    dflash_aux_layers_{1, 8, 16, 24, 31},  // 32 层模型：step=(32-2)/(5-1)=7.5 → {1,8.5,16,23.5,31} → {1,8,16,24,31}
    d_comm_(ctx.comm.d_comm),
    tune_layer_num_(model.tune_layer_num),
    is_warm_up_{*ctx.is_warm_up},
    ctx_(const_cast<Context*>(&ctx))
{
    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
    }

    attn_layer_ =
        std::make_unique<UnifiedAttentionLayer>(model, attn, engine, attn_tp_size_, ctx, phases, (bool)moe_ffn_layer_);

    if (std::find(model.layer_types.begin(), model.layer_types.end(), 1) != model.layer_types.end()) {
        linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(model, attn, engine, attn_tp_size_, ctx, phases);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }
}

void UnifiedDecoder::AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                              Tensor&       residual,
                                              const Tensor& bias,
                                              const Tensor& weight,
                                              int           token_num,
                                              int           group0,
                                              int           group1,
                                              const int*    local_token_nums)
{
    const auto dtype = hidden_states.dtype();

    const auto stream = core::Context::stream().handle();

    if (0) {}
    else if (group0 || group1) {
        d_comm_->AllreduceResidualBiasRMSnormEx(hidden_states.raw_data(),
                                                residual.data_or((void*)nullptr),
                                                bias.data_or((void*)nullptr),
                                                weight.raw_data(),
                                                rmsnorm_eps_,
                                                hidden_units_,
                                                dtype,
                                                group0,
                                                group1,
                                                local_token_nums,
                                                stream);
        sync_check_cuda_error();
    }
    else if (d_comm_) {
        d_comm_->AllreduceResidualBiasRMSnorm(hidden_states.raw_data(),
                                              residual.data_or((void*)nullptr),
                                              bias.data_or((void*)nullptr),
                                              weight.raw_data(),
                                              rmsnorm_eps_,
                                              hidden_units_,
                                              token_num,
                                              dtype,
                                              0,
                                              stream);
        sync_check_cuda_error();
    }
    else {
        invokeResidualBiasRMSNorm(hidden_states.raw_data(),
                                  residual.data_or((void*)nullptr),
                                  weight.raw_data(),
                                  bias.data_or((void*)nullptr),
                                  dtype,
                                  hidden_units_,
                                  token_num,
                                  rmsnorm_eps_,
                                  stream);
        sync_check_cuda_error();
    }
}

void UnifiedDecoder::Forward(int phase, TensorMap& args, const std::vector<WeightType*>& weights)
{
    /**
     * input tensors:
     *   \param decoder_input [token_num, hidden_units], float
     *   \param output_norm_weight [hidden_dims], float
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param last_token_hidden_units [batch_size, hidden_units]
     *   \param block_ptrs [total_block_counts], void*
     */

    constexpr auto device = kDEVICE;

    Tensor      local_residual   = args.try_consume("input_embeds");
    const auto& local_token_nums = args.at("batch").data<BatchData*>()[0]->local_token_num;

    const auto local_token_num  = local_residual.shape(0);
    const auto global_token_num = std::accumulate(local_token_nums.begin(), local_token_nums.end(), ssize_t{});

    TM_CHECK_EQ(local_token_num, local_token_nums[attn_dp_rank_]);

    const DataType dtype = local_residual.dtype();

    // DFlash 调试：打印 Forward 开始时的状态
    TM_LOG_INFO("[DFlash] Forward called: decoder=%p, enable_dflash_={}, dflash_draft_model_={}",
                (void*)this, enable_dflash_, (void*)dflash_draft_model_);

    // DFlash: 预先获取 selected_token_pos（如果存在）
    const Buffer* selected_pos_ptr = nullptr;
    if (enable_dflash_ && dflash_draft_model_) {
        if (args.try_("selected_token_pos")) {
            selected_pos_ptr = &args.at("selected_token_pos").buffer();
            TM_LOG_INFO("[DFlash] Found selected_token_pos, size={}", selected_pos_ptr->size());
        } else {
            TM_LOG_INFO("[DFlash] No selected_token_pos in args (phase={})", phase);
        }
        TM_LOG_INFO("[DFlash] DFlash enabled: enable={}, model={}, phase={}",
                    enable_dflash_, (void*)dflash_draft_model_, phase);
    } else {
        if (!enable_dflash_) {
            TM_LOG_INFO("[DFlash] NOT enabled: enable_dflash_={}", enable_dflash_);
        }
        if (!dflash_draft_model_) {
            TM_LOG_INFO("[DFlash] NO draft model: dflash_draft_model_={}", (void*)dflash_draft_model_);
        }
    }

    Tensor global_hidden_states;
    if (d_comm_) {
        Buffer symm_buf      = args.at("symm_buf").buffer();
        global_hidden_states = {symm_buf.view(dtype), {global_token_num, (int)hidden_units_}};
    }
    else {
        global_hidden_states = {{global_token_num, (int)hidden_units_}, local_residual.dtype(), kDEVICE};
    }

    Tensor local_hidden_states;
    if (attn_dp_size_ > 1) {  // Offset hidden states buffer for mixed DP
        TM_CHECK_EQ(local_token_nums.size(), attn_dp_size_);
        std::vector offsets(attn_dp_size_ + 1, 0);
        std::inclusive_scan(local_token_nums.data(), local_token_nums.data() + attn_dp_size_, offsets.begin() + 1);
        const int offset    = offsets[attn_dp_rank_];
        local_hidden_states = global_hidden_states.slice({offset, 0}, {local_token_num, -1});

        // dbg(attn_dp_size_, attn_dp_rank_, local_token_nums, local_token_num, global_token_num);
    }
    else {
        local_hidden_states = global_hidden_states;
    }

    TM_DEBUG_TENSOR(local_residual, "res", 1);
    TM_DEBUG_TENSOR(weights.at(0)->self_attn_norm, "norm_weight", 2);

    const auto stream = core::Context::stream().handle();

    invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->self_attn_norm, rmsnorm_eps_, stream);
    sync_check_cuda_error();

    TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);

    // auto stack_alloc{core::Context::device_alloc().adapt<core::StackAllocatorImpl>()};
    // core::ContextGuard ctx{Allocator{stack_alloc}};

    TM_LOG_INFO("[DFlash] Starting layer loop: layer_num_={}", (int)layer_num_);

    for (int layer = 0; layer < layer_num_; ++layer) {

        // stack_alloc->iter();

        if (global_token_num == 0) {
            break;
        }

        if (is_warm_up_ && layer >= tune_layer_num_) {
            continue;
        }

        /////////////////////////////////////////////
        /// self-attention or linear-attention
        if (weights.at(layer)->linear_attn_weights) {
            linear_attn_layer_->Forward(
                {phase, local_hidden_states, local_hidden_states, weights.at(layer)->linear_attn_weights.get(), layer});
        }
        else {
            attn_layer_->Forward(
                {phase, local_hidden_states, local_hidden_states, weights.at(layer)->self_attn_weights.get(), layer});
        }

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);

        // For gated delta networks, we may need a different output.bias name or it doesn't have it.
        // We will just use `output.bias` from either layer.
        Tensor out_bias;
        if (weights.at(layer)->linear_attn_weights) {
            out_bias = weights.at(layer)->linear_attn_weights->out_proj.bias;
        }
        else {
            out_bias = weights.at(layer)->self_attn_weights->output.bias;
        }

        AllreduceResidualRMSnorm(global_hidden_states,
                                 local_residual,
                                 out_bias,
                                 weights.at(layer)->ffn_norm,
                                 local_token_num,
                                 attn_tp_group_,
                                 0,
                                 local_token_nums.data());

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network

        std::optional<MoeFfnLayer::ForwardParam> moe_fwd_param;

        if (weights.at(layer)->moe_weights) {
            moe_fwd_param = MoeFfnLayer::ForwardParam{global_hidden_states,
                                                      global_hidden_states,
                                                      weights.at(layer)->moe_weights.get(),
                                                      ffn_layer_ ? 1.f : 0.f,
                                                      layer};
            moe_ffn_layer_->Forward(*moe_fwd_param);
        }

        if (weights.at(layer)->ffn_weights) {
            ffn_layer_->forward(
                {global_hidden_states, global_hidden_states, weights.at(layer)->ffn_weights.get(), (int)layer});
        }

        if (moe_fwd_param) {
            moe_ffn_layer_->Combine(*moe_fwd_param);
        }

        TM_DEBUG_TENSOR(global_hidden_states, Concat("ffn_block", layer), 2);

        // DFlash: 收集指定层的 hidden states
        if (enable_dflash_ && dflash_draft_model_) {
            for (int i = 0; i < 5; ++i) {
                if (layer == dflash_aux_layers_[i]) {
                    if (aux_hidden_states_.size() <= (size_t)i) {
                        aux_hidden_states_.resize(5);
                    }

                    // 简化：无论是否有 selected_token_pos，都收集全部 token
                    // 这样可以在 prefill 和 decode 阶段都能工作
                    size_t collect_size = global_token_num;

                    aux_hidden_states_[i] = Tensor{{(int)collect_size, (int)hidden_units_}, dtype, kDEVICE};

                    Copy(global_hidden_states, aux_hidden_states_[i]);
                    sync_check_cuda_error();  // Check for CUDA errors after copy
                    TM_LOG_INFO("[DFlash] Collected aux hidden state at layer {}, token_num={}, shape=[%zu, %d], hidden_units_={}, total_layers={}",
                                layer, collect_size, aux_hidden_states_[i].shape(0), (int)aux_hidden_states_[i].shape(1),
                                (int)hidden_units_, aux_hidden_states_.size());
                    break;
                }
            }
        }

        const bool last = layer == layer_num_ - 1;

        auto& scale_weight = !last ? weights.at(layer + 1)->self_attn_norm : args.at("output_norm_weight");

        AllreduceResidualRMSnorm(global_hidden_states,
                                 local_residual,
                                 {},
                                 scale_weight,
                                 local_token_num,
                                 0,
                                 attn_tp_group_,
                                 local_token_nums.data());
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_residual, Concat("residual1", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", layer + 1), 2);

        // if (layer == layer_num_ - 1) {
        //     args.at("batch").data<BatchData*>()[0]->Notify();
        // }
    }

    // Token indices selected for decoding
    const Buffer selected_pos = args.consume("selected_token_pos").buffer();
    // dbg(selected_pos);
    // When there are no prefill sequences, token selection is not needed
    const bool reuse_hidden_states = selected_pos.size() == local_token_num;

    const bool output_hidden_states = args.try_("output_hidden_states");

    Tensor hidden_states{local_hidden_states};

    if (d_comm_ && (output_hidden_states || reuse_hidden_states)) {
        // The full `hidden_states` buffer is needed for output but it's a ref into `symm_buf` atm.
        // Copy to residual buf so that `symm_buf` may be reused safely later
        Copy(hidden_states, local_residual);
        hidden_states = local_residual;
    }

    Tensor selected_states;
    if (reuse_hidden_states) {
        selected_states = hidden_states;
    }
    else {
        selected_states = {{selected_pos.size(), (int)hidden_units_}, dtype, kDEVICE};
        CollectHiddenStates(hidden_states, selected_pos, selected_states, stream);
    }
    args.produce("hidden_states", selected_states);

    // TM_DEBUG_TENSOR(selected_states.slice(0, selected_pos.size()), "out", 1);

    if (output_hidden_states) {
        args.produce("full_hidden_states", hidden_states);
    }

    // DFlash: aux_hidden_states 已在层循环中收集
    TM_LOG_INFO("[DFlash] Checking DFlash conditions: enable_dflash_={}, dflash_draft_model_={}, phase={}",
                enable_dflash_, (void*)dflash_draft_model_, phase);
    if (enable_dflash_ && dflash_draft_model_ && phase == 0) {
        TM_LOG_INFO("[DFlash] Phase={}: attempting speculative decoding", phase);
        TM_LOG_INFO("[DFlash] aux_hidden_states_.size()={}", aux_hidden_states_.size());

        // Log aux hidden state shapes before GenerateDraft
        for (size_t i = 0; i < aux_hidden_states_.size(); ++i) {
            const auto& t = aux_hidden_states_[i];
            TM_LOG_INFO("[DFlash] aux_hidden_states_[%zu] shape=[%d, %d] dtype=%d",
                       i, (int)t.shape(0), (int)t.shape(1), (int)t.dtype());
        }

        if (!aux_hidden_states_.empty()) {
            // Check draft model weights before running
            auto* draft_weight = dflash_draft_model_->GetDraftWeight();
            if (!draft_weight || !draft_weight->lm_head || !draft_weight->embed_tokens) {
                TM_LOG_WARNING("[DFlash] Draft weights not properly loaded, skipping DFlash");
                TM_LOG_WARNING("[DFlash]   draft_weight=%p, lm_head=%p, embed_tokens=%p",
                            (void*)draft_weight,
                            draft_weight ? (void*)draft_weight->lm_head.raw_data() : nullptr,
                            draft_weight ? (void*)draft_weight->embed_tokens.raw_data() : nullptr);
            } else {
                // Speculative decoding:
                // 1. Generate draft tokens from aux hidden states
                // 2. Verify against target logits
                // 3. Output accepted tokens to env for Generation to use

                Tensor draft_tokens, draft_logits;
                TM_LOG_INFO("[DFlash] Calling GenerateDraft NOW...");
                dflash_draft_model_->GenerateDraft(aux_hidden_states_, draft_tokens, draft_logits);
                TM_LOG_INFO("[DFlash] GenerateDraft COMPLETED");

                TM_LOG_INFO("[DFlash] GenerateDraft returned: draft_tokens={}, draft_logits={}",
                            (void*)draft_tokens.raw_data(), (void*)draft_logits.raw_data());

                if (draft_tokens && draft_tokens.shape(0) > 0) {
                    TM_LOG_INFO("[DFlash] Draft tokens generated: count={}", draft_tokens.shape(0));

                    Tensor* target_logits_ptr = args.try_("logits");
                    if (target_logits_ptr) {
                        Tensor target_logits = *target_logits_ptr;
                        TM_LOG_INFO("[DFlash] Target logits shape: [{}]", target_logits.shape(0));

                        Tensor accepted, mask;
                        dflash_draft_model_->VerifyDraft(draft_tokens, target_logits, accepted, mask);

                        TM_LOG_INFO("[DFlash] VerifyDraft: accepted={} tokens", accepted.shape(0));

                        // Output accepted tokens to env so Generation can use them
                        args.produce("dflash_accepted_tokens", accepted);
                        args.produce("dflash_accept_mask", mask);

                        TM_LOG_INFO("[DFlash] Success: {} accepted draft tokens output to Generation",
                                  accepted.shape(0));
                    } else {
                        TM_LOG_WARNING("[DFlash] No target logits found in args!");
                    }
                } else {
                    TM_LOG_WARNING("[DFlash] GenerateDraft returned empty tokens!");
                }
            }
        } else {
            TM_LOG_WARNING("[DFlash] aux_hidden_states_ is empty, cannot generate drafts!");
        }
    } else {
        if (enable_dflash_) {
            TM_LOG_INFO("[DFlash] Skipping: enable={}, model={}, phase={}",
                        enable_dflash_, (void*)dflash_draft_model_, phase);
        }
    }
}

}  // namespace turbomind
