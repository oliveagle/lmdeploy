

#pragma once

#include <memory>
#include <vector>

#include <cublas_v2.h>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/models/llama/DFlashDraftWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct EngineParam;
class Context;

/**
 * DFlash draft model for speculative decoding
 *
 * Uses target model's intermediate layer hidden states to generate
 * multiple draft tokens in parallel through a diffusion process.
 */
class DFlashDraftModel {
public:
    DFlashDraftModel(const ModelParam& model,
                     const EngineParam& engine,
                     const Context& ctx,
                     int num_spec_tokens = 8,
                     int num_draft_layers = 8);

    ~DFlashDraftModel();

    // Disable copy
    DFlashDraftModel(const DFlashDraftModel&) = delete;
    DFlashDraftModel& operator=(const DFlashDraftModel&) = delete;

    /**
     * Extract aux_hidden_states from target model layer outputs
     */
    void ExtractAuxHidden(const std::vector<Tensor>& layer_outputs,
                          std::vector<Tensor>& aux_states);

    /**
     * Generate draft tokens from aux hidden states
     * @param aux_states 5 layer hidden states from target model
     * @param draft_tokens Output: draft token IDs [num_spec]
     * @param draft_logits Output: draft logits [num_spec, vocab]
     */
    void GenerateDraft(const std::vector<Tensor>& aux_states,
                      Tensor& draft_tokens,
                      Tensor& draft_logits);

    /**
     * Verify draft tokens against target model logits
     */
    void VerifyDraft(const Tensor& draft_tokens,
                    const Tensor& target_logits,
                    Tensor& accepted_tokens,
                    Tensor& accept_mask);

    // Getters
    int GetNumSpecTokens() const { return num_spec_tokens_; }
    int GetNumAuxLayers() const { return num_aux_layers_; }

    // Set draft weight pointer - use the weights from LlamaWeight directly
    void SetDraftWeightPointer(DFlashDraftWeight* weight) {
        if (weight) {
            weight_.reset();  // Release our own weight
            external_weight_ = weight;  // Use external weight
        }
    }

    // Get the current weight (external or internal)
    DFlashDraftWeight* GetDraftWeight() const {
        return external_weight_ ? external_weight_ : weight_.get();
    }

private:
    DFlashDraftWeight* external_weight_ = nullptr;  // External weight (from LlamaWeight)
    std::unique_ptr<DFlashDraftWeight> weight_;     // Internal weight
    cublasHandle_t cublas_ = nullptr;

    // Config
    const int hidden_size_;
    const int num_draft_layers_;
    const int num_aux_layers_;
    const int num_spec_tokens_;
    const int target_layer_ids_[5];
    const int vocab_size_;
};

}  // namespace turbomind
