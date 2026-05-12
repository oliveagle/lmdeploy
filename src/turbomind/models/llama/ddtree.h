#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <queue>
#include <algorithm>
#include <limits>

#include "src/turbomind/core/tensor.h"

using Tensor = turbomind::core::Tensor;

namespace turbomind {

/**
 * DDTree (Dense Depth Tree) structure for speculative decoding.
 *
 * A tree structure that allows parallel verification of multiple candidate
 * token sequences using ancestor-only attention masks. This gives ~30%
 * speedup over linear verification.
 */
struct DDTree {
    int n_nodes = 0;                  // Number of non-root nodes
    std::vector<int32_t> token_ids;   // Token IDs for each node (size n_nodes)
    std::vector<int> depths;          // Depth of each node (size n_nodes)
    std::vector<int> parents;         // Parent index for each node (size n_nodes+1, root=-1)
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // Child token to index (size n_nodes+1)
    std::vector<uint8_t> visibility;  // Visibility mask: (1+n_nodes)^2 row-major

    // Reset to empty state
    void clear() {
        n_nodes = 0;
        token_ids.clear();
        depths.clear();
        parents.clear();
        child_maps.clear();
        visibility.clear();
    }

    // Check if tree is empty
    bool empty() const {
        return n_nodes == 0;
    }

    // Get total nodes including root
    int total_nodes() const {
        return 1 + n_nodes;
    }
};

/**
 * DDTree builder.
 *
 * Builds a best-first tree from per-position top-K distributions.
 */
class DDTreeBuilder {
public:
    /**
     * Build a DDTree from draft logits.
     *
     * @param draft_logits     [num_draft, vocab_size] draft logits
     * @param num_draft        Number of draft positions
     * @param vocab_size       Vocabulary size
     * @param k                Top-K per position (default: 4)
     * @param budget           Maximum tree nodes (default: 22)
     * @param temperature      Temperature for logit scaling (default: 1.0)
     * @param chain_seed       Pre-seed with full top-1 chain (default: true)
     */
    static DDTree build(const Tensor& draft_logits,
                       int num_draft,
                       int vocab_size,
                       int k = 4,
                       int budget = 22,
                       float temperature = 1.0f,
                       bool chain_seed = true);

    /**
     * Build a DDTree from precomputed top-K log-probs and tokens.
     *
     * @param top_log_probs   [num_draft, k] log-probabilities
     * @param top_token_ids   [num_draft, k] token IDs
     * @param num_draft       Number of draft positions
     * @param k               Top-K per position
     * @param budget          Maximum tree nodes
     * @param chain_seed      Pre-seed with full top-1 chain
     */
    static DDTree build_from_topk(const float* top_log_probs,
                                const int32_t* top_token_ids,
                                int num_draft,
                                int k,
                                int budget,
                                bool chain_seed = true);

    /**
     * Extract top-K log-probabilities and tokens from logits.
     *
     * @param logits         [n_positions, vocab_size] logits
     * @param n_positions    Number of positions
     * @param vocab          Vocabulary size
     * @param k              Top-K
     * @param out_log_probs  [n_positions, k] output log-probs
     * @param out_token_ids  [n_positions, k] output token IDs
     * @param temperature    Temperature for scaling (default: 1.0)
     */
    static void extract_topk(const float* logits,
                           int n_positions,
                           int vocab,
                           int k,
                           float* out_log_probs,
                           int32_t* out_token_ids,
                           float temperature = 1.0f);
};

/**
 * DDTree verifier.
 *
 * Verifies a DDTree against target model logits and finds the
 * longest accepted token sequence.
 */
class DDTreeVerifier {
public:
    /**
     * Follow the verified tree and find accepted tokens.
     *
     * @param tree                DDTree structure
     * @param posterior_tokens    [total_nodes] target argmax tokens (posterior)
     * @param accepted_indices    Output: accepted node indices (including root)
     * @param out_next_token      Output: next bonus token from target
     * @return Number of accepted tokens (excluding root)
     */
    static int follow(const DDTree& tree,
                    const int32_t* posterior_tokens,
                    std::vector<int>& accepted_indices,
                    int32_t& out_next_token);

    /**
     * Build ancestor-only attention mask for tree verification.
     *
     * @param tree          DDTree structure
     * @param past_length   Past KV cache length
     * @param out_mask      Output: [kv_pad, q_pad] attention mask
     * @param kq_pad        Padding alignment for KQ (default: 32)
     */
    static void build_attention_mask(const DDTree& tree,
                                   int past_length,
                                   std::vector<uint16_t>& out_mask,
                                   int kq_pad = 32);

    /**
     * Build flat tree positions array for positional encoding.
     *
     * @param tree          DDTree structure
     * @param past_length   Past KV cache length
     * @param out_positions Output: [total_nodes] positions
     */
    static void build_positions(const DDTree& tree,
                              int past_length,
                              std::vector<int32_t>& out_positions);

    /**
     * Build parent IDs array for tree structure.
     *
     * @param tree          DDTree structure
     * @param out_parent_ids Output: [total_nodes] parent indices
     */
    static void build_parent_ids(const DDTree& tree,
                               std::vector<int32_t>& out_parent_ids);
};

}  // namespace turbomind
