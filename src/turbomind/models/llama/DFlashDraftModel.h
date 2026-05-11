#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include <cublas_v2.h>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/models/llama/DFlashDraftWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct EngineParam;
class Context;

/**
 * Cache entry for DFlash prefix caching (STORY-010: Advanced Features)
 */
struct DFlashCacheEntry {
    std::vector<int> tokens;       // Prompt tokens for this entry
    Tensor cached_ctx_k;          // Cached context K
    Tensor cached_ctx_v;          // Cached context V
    std::vector<Tensor> cached_aux_states;  // Cached aux hidden states
    uint64_t last_accessed;       // Timestamp for LRU
    size_t hash;                  // Hash of the tokens
};

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
     * @param input_tokens Optional: Input token sequence for prefix caching
     */
    void GenerateDraft(const std::vector<Tensor>& aux_states,
                    Tensor& draft_tokens,
                    Tensor& draft_logits,
                    const std::vector<int>* input_tokens = nullptr);

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

    // ========== Prefix Cache API (STORY-010: Advanced Features) ==========
    /**
     * Enable/disable prefix caching
     */
    void SetPrefixCacheEnabled(bool enabled) {
        enable_prefix_cache_ = enabled;
    }

    /**
     * Check if prefix caching is enabled
     */
    bool IsPrefixCacheEnabled() const {
        return enable_prefix_cache_;
    }

    /**
     * Clear the prefix cache
     */
    void ClearPrefixCache() {
        prefix_cache_.clear();
        cache_order_.clear();
    }

    /**
     * Get prefix cache size
     */
    size_t GetPrefixCacheSize() const {
        return prefix_cache_.size();
    }

    /**
     * Set maximum prefix cache entries
     */
    void SetMaxPrefixCacheEntries(size_t max_entries) {
        max_prefix_cache_entries_ = max_entries;
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

    // ========== Prefix Cache (STORY-010) ==========
    bool enable_prefix_cache_ = false;
    size_t max_prefix_cache_entries_ = 8;  // Max cache entries
    std::unordered_map<size_t, DFlashCacheEntry> prefix_cache_;  // Hash -> Entry
    std::vector<size_t> cache_order_;  // LRU order

    /**
     * Compute hash for input tokens
     */
    size_t ComputeTokensHash(const std::vector<int>& tokens) const;

    /**
     * Find matching prefix cache entry
     * Returns: matching entry if found, null otherwise
     */
    const DFlashCacheEntry* FindPrefixMatch(const std::vector<int>& tokens) const;

    /**
     * Store entry in prefix cache (with LRU eviction)
     */
    void StoreInPrefixCache(const std::vector<int>& tokens,
                        const Tensor& ctx_k,
                        const Tensor& ctx_v,
                        const std::vector<Tensor>& aux_states);

    /**
     * Evict least recently used entry
     */
    void EvictLRUEntry();
};

}  // namespace turbomind
