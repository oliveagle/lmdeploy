
#include <memory>
#include <vector>

#include "src/turbomind/generation/generation.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/generation/guided_decoding.h"
#include "src/turbomind/generation/logits_processor.h"
#include "src/turbomind/generation/sampling.h"
#include "src/turbomind/generation/stop_criteria.h"

#include "src/turbomind/kernels/sampling_topk_kernels.h"  // InitializeRandomStates

#include "src/turbomind/models/llama/llama_kernels.h"  // invokePadLastTokenIds
#include "src/turbomind/models/llama/DFlashDraftModel.h"
#include "src/turbomind/models/llama/dflash_kernels.h"
#include "src/turbomind/core/logger.h"

// #include "dbg.h"

namespace turbomind {

// 简化版 DFlash 验证函数声明（在 DFlashDraftModel.cu 中定义）
extern void DFlashVerifyDraftGPU(
    const Tensor& draft_tokens,
    const Tensor& target_logits,
    Tensor& accepted_tokens,
    Tensor& accept_mask,
    Tensor& max_logits,
    Tensor& accepted_count,  // Output: [1] device tensor for count
    int num_spec_tokens);

using std::unique_ptr;
using std::shared_ptr;
using std::vector;

struct GenerationData {
    Buffer_<uint8_t>  random_state;
    Buffer_<uint64_t> random_seed;
    Buffer_<bool>     random_init;
    Buffer_<int>      max_seq_len;
    Buffer_<int*>     token_ids_ptrs;
    Buffer_<int>      output_ids;

    bool random_init_needed;
    int  generation_size;
};

struct Generation::Impl {

    // child modules
    unique_ptr<LogitsProcessor> logits_processor_;
    unique_ptr<Sampling>        sampling_;
    shared_ptr<StopCriteria>    stop_criteria_;
    unique_ptr<GuidedDecoding>  guided_decoding_;

    // persistent
    Tensor_<int> token_ids_;

    // scheduling states
    vector<int*> h_token_ids_ptrs_;
    vector<int*> h_token_ids_free_;

    // execution states
    State random_state_;

    // immutable states
    Buffer_<int> output_ids_;

    std::vector<std::unique_ptr<GenerationData>> data_;

    // staging buffers
    Buffer_<uint8_t>  random_state_buf_;
    Buffer_<uint64_t> random_seed_buf_;
    Buffer_<bool>     random_init_buf_;
    Buffer_<int*>     token_ids_ptrs_buf_;
    Buffer_<int>      token_ids_buf_;
    Buffer_<int>      output_ids_buf_;

    const int max_batch_size_;
    const int session_len_;

    // DFlash statistics
    int dflash_total_draft_steps_{0};
    int dflash_total_draft_tokens_{0};
    int dflash_total_accepted_tokens_{0};
    int dflash_total_rejected_tokens_{0};
    int dflash_summary_interval_{10};  // Log summary every N draft steps

    // DFlash draft token storage (跨调用持久化)
    Buffer_<int> dflash_stored_draft_tokens_;  // 存储待验证的 draft tokens
    Buffer_<int> dflash_stored_draft_logits_;  // 存储 draft logits

    Impl(DataType              dtype,
         int                   max_batch_size,
         int                   session_len,
         int                   vocab_size,
         int                   vocab_size_padded,
         const comm::HostComm& tp_group,
         int                   phases):
        max_batch_size_{max_batch_size}, session_len_{session_len}
    {
        TM_CHECK_EQ(dtype, kFloat32);
        BaseGenerationParam base{max_batch_size, vocab_size, vocab_size_padded};
        logits_processor_ = std::make_unique<LogitsProcessor>(base, phases);
        sampling_         = std::make_unique<Sampling>(base, phases);
        stop_criteria_    = std::make_unique<StopCriteria>(base, phases);
        guided_decoding_  = std::make_unique<GuidedDecoding>(base, tp_group, phases);

        static_assert(sizeof(curandState_t) % alignof(curandState_t) == 0);
        random_state_ = {{max_batch_size_, (int)sizeof(curandState_t)}, kUint8, kDEVICE};
        token_ids_    = {{max_batch_size_, session_len_}, kDEVICE};
        output_ids_   = {max_batch_size_, kDEVICE};
        for (int i = 0; i < max_batch_size_; ++i) {
            h_token_ids_free_.push_back(token_ids_.data() + i * token_ids_.stride(0));
        }
        h_token_ids_ptrs_.resize(max_batch_size_);

        random_state_buf_ = {max_batch_size_ * (int)sizeof(curandState_t), kCPUpinned};
        random_seed_buf_  = {max_batch_size_, kCPUpinned};
        random_init_buf_  = {max_batch_size_, kCPUpinned};

        token_ids_ptrs_buf_ = {max_batch_size_, kCPUpinned};
        token_ids_buf_      = {max_batch_size_ * (ssize_t)session_len_, kCPUpinned};

        output_ids_buf_ = {max_batch_size_, kCPUpinned};

        for (int i = 0; i < phases; ++i) {
            auto d = std::make_unique<GenerationData>();

            d->random_state   = empty_like(random_state_buf_, kDEVICE);
            d->random_seed    = empty_like(random_seed_buf_, kDEVICE);
            d->random_init    = empty_like(random_init_buf_, kDEVICE);
            d->token_ids_ptrs = empty_like(token_ids_ptrs_buf_, kDEVICE);
            d->output_ids     = empty_like(output_ids_, kDEVICE);

            data_.push_back(std::move(d));
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        auto& d = *data_.at(phase);

        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        const auto& rc = b.rc;

        // random states
        d.random_init_needed = false;
        for (int i = 0; i < b.perm.size(); ++i) {
            const auto& c = *rc[i];
            if (TM_LIKELY(b.perm[i] < b.bs0)) {  // existing
                random_init_buf_[i] = false;
            }
            else if (c.random_state) {  // already initialized
                std::copy_n(
                    c.random_state, sizeof(curandState_t), random_state_buf_.data() + i * sizeof(curandState_t));
            }
            else {  // uninitialized
                d.random_init_needed = true;
                random_init_buf_[i]  = true;
                random_seed_buf_[i]  = rc[i]->gen_cfg.random_seed;
            }
        }
        copy(random_state_buf_, b.bsz, d.random_state);
        if (d.random_init_needed) {
            copy(random_init_buf_, b.bsz, d.random_init);
            copy(random_seed_buf_, b.bsz, d.random_seed);
        }

        vector<int> used(b.bs0);
        for (int i = 0; i < b.bsz; ++i) {
            if (b.perm[i] < b.bs0) {
                used[b.perm[i]] = 1;
            }
        }
        for (int i = 0; i < b.bs0; ++i) {
            if (!used[i]) {  // free unused chunks
                h_token_ids_free_.push_back(h_token_ids_ptrs_[i]);
            }
        }
        // swap-in token_ids
        int* token_ids_buf = token_ids_buf_.data();
        for (int i = 0; i < rc.size(); ++i) {
            if (const auto& c = *rc[i]; TM_UNLIKELY(b.perm[i] >= b.bs0)) {
                // allocation
                TM_CHECK(!h_token_ids_free_.empty());
                token_ids_ptrs_buf_[i] = h_token_ids_free_.back();
                h_token_ids_free_.pop_back();
                // copy to staging buffer
                std::copy_n(c.token_ids, c.seq_len, token_ids_buf);
                copy(token_ids_buf, c.seq_len, token_ids_ptrs_buf_[i]);
                token_ids_buf += c.seq_len;
            }
            else {
                token_ids_ptrs_buf_[i] = h_token_ids_ptrs_[b.perm[i]];
            }
        }

        copy(token_ids_ptrs_buf_, b.bsz, d.token_ids_ptrs);

        // update `h_token_ids_ptrs_`
        std::copy_n(token_ids_ptrs_buf_.data(), b.bsz, h_token_ids_ptrs_.data());

        d.generation_size = 0;
        for (int i = 0; i < rc.size(); ++i) {
            const auto& c = *rc[i];
            d.generation_size += c.generating;
        }
        // dbg(d.generation_size);
        TM_LOG_ERROR("[DFlash] Generation Setup: generation_size=%d, batch_size=%d", (int)d.generation_size, (int)rc.size());

        logits_processor_->Setup(phase, env);
        sampling_->Setup(phase, env);
        stop_criteria_->Setup(phase, env);
        guided_decoding_->Setup(phase, env);
    }

    void Prepare(int phase, TensorMap& env)
    {
        auto& d = *data_.at(phase);

        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        if (auto g = copy.group()) {
            Warp(random_state_.front(), d.random_state, b.bs0, b.perm, random_state_.back(), copy);
            random_state_.Swap();
        }
    }

    void Unprep(int phase, TensorMap& env)
    {
        auto& d    = *data_.at(phase);
        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        // state -> data
        copy(random_state_.front().buffer(), b.bsz * sizeof(curandState_t), d.random_state);
        copy(output_ids_, b.bsz, d.output_ids);
    }

    void Fetch(int phase, TensorMap& env)
    {
        auto& d    = *data_.at(phase);
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        copy(d.random_state, d.random_state.size(), random_state_buf_);
        env.produce("random_state", random_state_buf_);

        copy(d.output_ids, d.output_ids.size(), output_ids_buf_);
        env.produce("output_ids", output_ids_buf_);

        sampling_->Fetch(phase, env);
    }

    void Update(int phase, TensorMap& env)
    {
        sampling_->Update(phase, env);
    }

    void Forward(int phase, TensorMap& env)
    {
        TM_LOG_INFO("[DFlash] Generation::Forward ENTRY: phase={}", phase);
        auto& d = *data_.at(phase);
        auto& b = *env.at("batch").data<BatchData*>()[0];

        const auto stream = core::Context::stream().handle();

        if (d.random_init_needed) {
            InitializeRandomStates((curandState_t*)random_state_.front().raw_data(),
                                   d.random_seed.data(),
                                   d.random_init.data(),
                                   b.bsz,
                                   stream);
            sync_check_cuda_error();
        }

        env.emplace("output_ids", output_ids_);              // out
        env.emplace("curand_state", random_state_.front());  // inout

        if (const int gs = d.generation_size) {
            env.emplace("token_ids_ptrs", d.token_ids_ptrs.slice(0, gs));

            auto logits = env.consume("logits");

            if (logits.dtype() != kFloat32) {
                auto tmp = empty_like(logits, kFloat32);
                invokeCastFloat2D(logits, tmp, stream);
                logits = std::move(tmp);
            }

            env.produce("logits", logits.slice(0, gs));

            Buffer_<int> output_pos{max_batch_size_, kDEVICE};
            Copy(env.at("sequence_length").buffer(), gs, output_pos);

            logits_processor_->Forward(phase, env);

            guided_decoding_->FillMask(phase, env);
            guided_decoding_->ApplyMask(phase, env);

            /// DFlash speculative decoding: check for pending draft tokens to verify
            fprintf(stderr, "[DFlash] Generation Forward: checking dflash_pending_draft_tokens, contains=%d, generation_size=%d\n",
                       (int)env.contains("dflash_pending_draft_tokens"), gs);
            if (env.contains("dflash_pending_draft_tokens") && env.contains("logits")) {
                const auto& pending_draft = env.at("dflash_pending_draft_tokens");

                // Safety check: only proceed if we have valid draft tokens
                if (pending_draft.shape(0) <= 0) {
                    TM_LOG_DEBUG("[DFlash] No pending draft tokens (shape={}), skipping verification", pending_draft.shape(0));
                    sampling_->Forward(phase, env);
                } else {
                    const auto& target_logits = env.at("logits");

                    TM_LOG_INFO("[DFlash] Generation: Verifying first draft token against target logits (draft_count=%d)",
                               pending_draft.shape(0));

                    // 获取 draft model（从 env 中）
                    // 注意：我们需要通过 Engine 或 LanguageModel 传递 draft model
                    // 临时方案：在 Generation 内部实现简单的验证逻辑
                    // 这个逻辑基于 DFlashVerifyDraftGPU，但不需要 draft model

                    // DFlash verification: only verify first draft token against target model
                    // In decode mode, target_logits has shape [1, vocab_size] (prediction for next token)
                    // pending_draft has shape [num_spec] (predictions for future tokens)
                    // We can only verify the first draft token since target model predicts 1 token

                    const int num_spec = pending_draft.shape(0);
                    const int vocab_size = target_logits.shape(1);

                    TM_LOG_DEBUG("[DFlash] Verify: num_spec=%d, vocab_size=%d", num_spec, vocab_size);

                    // Allocate output tensors for verification
                    Tensor accepted_tokens, accept_mask, max_logits;
                    Buffer_<int> h_accepted_count{1, kCPUpinned};
                    Tensor accepted_count = Tensor(h_accepted_count.data(), {1}, kCPU);

                    DFlashVerifyDraftGPU(
                        pending_draft,
                        target_logits,
                        accepted_tokens,
                        accept_mask,
                        max_logits,
                        accepted_count,
                        num_spec);

                    // Sync stream and copy count to host
                    const auto verify_stream = core::Context::stream().handle();
                    cudaStreamSynchronize(verify_stream);

                    int accepted_count_val = h_accepted_count[0];
                    dflash_total_accepted_tokens_ += accepted_count_val;
                    dflash_total_rejected_tokens_ += (num_spec - accepted_count_val);

                    TM_LOG_INFO("[DFlash] Verification complete: %d/%d accepted (%.1f%%)",
                               accepted_count_val, num_spec,
                               100.0f * accepted_count_val / num_spec);

                    // Sample from target logits (for rejected positions or if all rejected)
                    sampling_->Forward(phase, env);

                    dflash_total_draft_steps_++;
                    dflash_total_draft_tokens_ += num_spec;
                    // Count will be available after sync; update stats in a follow-up step

                    TM_LOG_INFO("[DFlash] Verification complete: %d draft tokens checked", num_spec);

                    // Periodic summary logging
                    if (dflash_total_draft_steps_ % dflash_summary_interval_ == 0 && dflash_total_draft_steps_ > 0) {
                        float overall_accept_rate = dflash_total_draft_tokens_ > 0
                            ? (float)dflash_total_accepted_tokens_ / dflash_total_draft_tokens_ * 100.0f : 0.0f;
                        TM_LOG_INFO("[DFlash] ================ STATS SUMMARY ================");
                        TM_LOG_INFO("[DFlash] Draft Steps:   %d", dflash_total_draft_steps_);
                        TM_LOG_INFO("[DFlash] Draft Tokens:  %d", dflash_total_draft_tokens_);
                        TM_LOG_INFO("[DFlash] Accepted:      %d (%.1f%%)", dflash_total_accepted_tokens_, overall_accept_rate);
                        TM_LOG_INFO("[DFlash] Rejected:      %d", dflash_total_rejected_tokens_);
                        TM_LOG_INFO("[DFlash] =================================================");
                    }
                }
            } else {
                // Normal sampling (no DFlash or no pending draft tokens)
                sampling_->Forward(phase, env);
            }

            guided_decoding_->Update(phase, env);

            AppendTokenIds(d.token_ids_ptrs.data(), output_ids_.data(), output_pos.data(), gs, stream);

            stop_criteria_->Forward(phase, env);
        }
    }
};

Generation::~Generation() = default;

Generation::Generation(DataType              dtype,
                       int                   max_batch_size,
                       int                   session_len,
                       int                   vocab_size,
                       int                   vocab_size_padded,
                       const comm::HostComm& tp_group,
                       int                   phases):
    impl_{std::make_unique<Impl>(dtype, max_batch_size, session_len, vocab_size, vocab_size_padded, tp_group, phases)}
{
}

void Generation::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kSetup) {
        return impl_->Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        return impl_->Prepare(phase, env);
    }
    else if (op == BatchOp::kForward) {
        return impl_->Forward(phase, env);
    }
    else if (op == BatchOp::kUnprep) {
        return impl_->Unprep(phase, env);
    }
    else if (op == BatchOp::kFetch) {
        return impl_->Fetch(phase, env);
    }
    else if (op == BatchOp::kUpdate) {
        return impl_->Update(phase, env);
    }
}

void Generation::GetDFlashStats(int& total_draft_steps, int& total_draft_tokens, int& total_accepted_tokens, int& total_rejected_tokens) const
{
    total_draft_steps = impl_->dflash_total_draft_steps_;
    total_draft_tokens = impl_->dflash_total_draft_tokens_;
    total_accepted_tokens = impl_->dflash_total_accepted_tokens_;
    total_rejected_tokens = impl_->dflash_total_rejected_tokens_;
}

}  // namespace turbomind
