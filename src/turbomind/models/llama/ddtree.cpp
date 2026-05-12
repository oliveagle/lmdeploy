#include "src/turbomind/models/llama/ddtree.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "src/turbomind/core/tensor.h"

namespace turbomind {

// Helper: align up to nearest multiple
static inline int align_up(int x, int a) {
    return ((x + a - 1) / a) * a;
}

// F16 encoding for 0 and -inf
static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

// ============================================================================
// DDTreeBuilder Implementation
// ============================================================================

void DDTreeBuilder::extract_topk(const float* logits,
                               int n_positions,
                               int vocab,
                               int k,
                               float* out_log_probs,
                               int32_t* out_token_ids,
                               float temperature) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry& a, const Entry& b) {
        return a.logit > b.logit;
    };

    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    for (int i = 0; i < n_positions; i++) {
        const float* li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(k);

        // Online log-sum-exp with running max
        float running_max = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            // Online logsumexp
            if (l > running_max) {
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            // Top-K maintenance using min-heap
            if ((int)heap.size() < k) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);

        // Sort the K entries descending
        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k_idx = 0; k_idx < k; k_idx++) {
            out_log_probs[(size_t)i * k + k_idx] = heap[k_idx].logit - log_z;
            out_token_ids[(size_t)i * k + k_idx] = heap[k_idx].id;
        }
    }
}

DDTree DDTreeBuilder::build_from_topk(const float* top_log_probs,
                                    const int32_t* top_token_ids,
                                    int num_draft,
                                    int k,
                                    int budget,
                                    bool chain_seed) {
    DDTree tree;
    if (budget <= 0 || num_draft <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    // Heap entry: neg_logw, ranks, parent_index, depth, rank, logw
    struct HeapEntry {
        float neg_logw;
        std::vector<int> ranks;
        int parent_index;
        int depth;
        int rank;
        float logw;
    };

    struct HeapCmp {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const {
            // std::priority_queue is a max-heap; we want smallest neg_logw at top
            return a.neg_logw > b.neg_logw;
        }
    };

    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);
    tree.parents.push_back(-1);  // root
    tree.child_maps.emplace_back();  // root's children

    if (chain_seed) {
        // Pre-seed full top-1 chain
        const int chain_depth = std::min(num_draft, budget);
        float cum_logw = 0.0f;
        int prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * k + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * k + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (k > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * k + 0]
                    + top_log_probs[(size_t)(d - 1) * k + 1];
                heap.push({
                    /*neg_logw*/ -sibling_logw,
                    /*ranks   */ {1},
                    /*parent  */ prev_idx,
                    /*depth   */ d,
                    /*rank    */ 1,
                    /*logw    */ sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        // Paper-style pure best-first: seed with depth-1 top-1
        const float root_logw = top_log_probs[0 * k + 0];
        heap.push({
            /*neg_logw*/ -root_logw,
            /*ranks   */ {0},
            /*parent  */ 0,
            /*depth   */ 1,
            /*rank    */ 0,
            /*logw    */ root_logw,
        });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int depth_minus_1 = top.depth - 1;
        const int rank = top.rank;
        const int32_t token_id = top_token_ids[(size_t)depth_minus_1 * k + rank];

        const int current_index = tree.n_nodes + 1;
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        // Push next sibling
        if (rank + 1 < k) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * k + rank]
                + top_log_probs[(size_t)depth_minus_1 * k + rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({
                /*neg_logw*/ -sibling_logw,
                /*ranks   */ std::move(sibling_ranks),
                /*parent  */ top.parent_index,
                /*depth   */ top.depth,
                /*rank    */ rank + 1,
                /*logw    */ sibling_logw,
            });
        }

        // Push first child
        if (top.depth < num_draft) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth * k + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({
                /*neg_logw*/ -child_logw,
                /*ranks   */ std::move(child_ranks),
                /*parent  */ current_index,
                /*depth   */ top.depth + 1,
                /*rank    */ 0,
                /*logw    */ child_logw,
            });
        }
    }

    // Build ancestor-only visibility mask
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0 * N + 0] = 1;  // root sees itself
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];
        // Inherit parent's visibility
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }

    return tree;
}

DDTree DDTreeBuilder::build(const Tensor& draft_logits,
                          int num_draft,
                          int vocab_size,
                          int k,
                          int budget,
                          float temperature,
                          bool chain_seed) {
    // Convert draft_logits to CPU float32 if needed
    // For now, assume we have CPU logits (we'll add GPU support later)
    const float* logits_ptr = static_cast<const float*>(draft_logits.raw_data());

    // Extract top-K
    std::vector<float> top_log_probs((size_t)num_draft * k);
    std::vector<int32_t> top_token_ids((size_t)num_draft * k);
    extract_topk(logits_ptr, num_draft, vocab_size, k,
                 top_log_probs.data(), top_token_ids.data(), temperature);

    // Build tree
    return build_from_topk(top_log_probs.data(), top_token_ids.data(),
                          num_draft, k, budget, chain_seed);
}

// ============================================================================
// DDTreeVerifier Implementation
// ============================================================================

int DDTreeVerifier::follow(const DDTree& tree,
                         const int32_t* posterior_tokens,
                         std::vector<int>& accepted_indices,
                         int32_t& out_next_token) {
    accepted_indices.clear();
    accepted_indices.reserve(tree.n_nodes + 1);
    accepted_indices.push_back(0);  // root

    int current_index = 0;
    int next_token = posterior_tokens[current_index];

    while (true) {
        const auto& children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) {
            break;
        }
        current_index = it->second;
        accepted_indices.push_back(current_index);
        next_token = posterior_tokens[current_index];
    }

    out_next_token = next_token;
    return (int)accepted_indices.size() - 1;  // exclude root
}

void DDTreeVerifier::build_attention_mask(const DDTree& tree,
                                        int past_length,
                                        std::vector<uint16_t>& out_mask,
                                        int kq_pad) {
    const int N = 1 + tree.n_nodes;
    const int kv_len = past_length + N;
    const int kv_pad = align_up(kv_len, kq_pad);
    const int q_pad = align_up(N, kq_pad);

    out_mask.assign((size_t)kv_pad * q_pad, F16_NEG_INF);

    for (int q = 0; q < N; q++) {
        // Past KV cache - attend freely
        for (int k = 0; k < past_length; k++) {
            out_mask[(size_t)q * kv_pad + k] = F16_ZERO;
        }
        // Tree nodes - attend only to ancestors
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                out_mask[(size_t)q * kv_pad + past_length + j] = F16_ZERO;
            }
        }
    }
}

void DDTreeVerifier::build_positions(const DDTree& tree,
                                   int past_length,
                                   std::vector<int32_t>& out_positions) {
    const int N = 1 + tree.n_nodes;
    out_positions.resize(N);
    out_positions[0] = past_length;  // root at past_length

    for (int i = 1; i < N; i++) {
        // Position is past_length + depth - 1 (depth starts at 1)
        out_positions[i] = past_length + tree.depths[i - 1] - 1;
    }
}

void DDTreeVerifier::build_parent_ids(const DDTree& tree,
                                    std::vector<int32_t>& out_parent_ids) {
    const int N = 1 + tree.n_nodes;
    out_parent_ids.resize(N);

    for (int i = 0; i < N; i++) {
        out_parent_ids[i] = tree.parents[i];
    }
}

}  // namespace turbomind
