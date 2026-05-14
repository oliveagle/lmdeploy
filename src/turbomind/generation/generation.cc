
#include <memory>

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

// #include "dbg.h"

namespace turbomind {

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

            // DFlash speculative decoding: check if we have pre-verified tokens
            if (env.contains("dflash_accepted_tokens")) {
                // Use DFlash accepted tokens instead of sampling
                const auto& dflash_accepted = env.at("dflash_accepted_tokens");
                const auto& dflash_mask = env.at("dflash_accept_mask");

                // Copy accepted tokens to output_ids
                // TODO: need to handle multiple batches properly
                int num_accepted = dflash_accepted.shape(0);
                Copy(dflash_accepted.buffer(), num_accepted, output_ids_);

                // Update DFlash statistics
                dflash_total_accepted_tokens_ += num_accepted;
                dflash_total_draft_steps_ += 1;
                // Assuming 8 speculative tokens per step
                dflash_total_draft_tokens_ += 8;

                TM_LOG_INFO("[DFlash] Using %d accepted tokens from DFlash (stats: steps=%d, accepted=%d, rejected=%d)",
                           num_accepted, dflash_total_draft_steps_, dflash_total_accepted_tokens_,
                           8 - num_accepted);

                // Update token_ids_ptrs
                AppendTokenIds(d.token_ids_ptrs.data(), output_ids_.data(), output_pos.data(), num_accepted, stream);

                // For rejected tokens, still need to sample
                int num_rejected = dflash_mask.shape(0) - num_accepted;
                if (num_rejected > 0) {
                    dflash_total_rejected_tokens_ += num_rejected;
                    TM_LOG_INFO("[DFlash] Sampling %d rejected tokens (stats: rejected=%d, total_rejected=%d)",
                               num_rejected, dflash_total_rejected_tokens_);
                    sampling_->Forward(phase, env);
                }
            } else {
                // Normal sampling
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
