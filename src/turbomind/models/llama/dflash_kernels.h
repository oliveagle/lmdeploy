

#pragma once

#include <cuda_fp16.h>

#include "src/turbomind/core/tensor.h"

namespace turbomind {

/**
 * DFlash CUDA kernels (FP16)
 *
 * All kernels operate on FP16 (half) data with FP32 accumulation
 * for numerical stability.
 */

void DFlashAttentionKernel(const void* query,
                           const void* key_ctx,
                           const void* key_draft,
                           const void* value_ctx,
                           const void* value_draft,
                           void* output,
                           int num_draft,
                           int num_ctx,
                           int hidden_size,
                           int head_dim,
                           cudaStream_t stream);

void DFlashMLPKernel(const void* hidden,
                     const void* gate,
                     const void* up,
                     const void* down,
                     void* output,
                     int num_draft,
                     int hidden_size,
                     int intermediate_size,
                     cudaStream_t stream);

void DFlashForwardKernel(const void* aux_hidden,
                         const void* qkv_proj,
                         const void* o_proj,
                         const void* gate_up_proj,
                         const void* down_proj,
                         const void* input_ln,
                         const void* post_ln,
                         void* output,
                         int num_layers,
                         int hidden_size,
                         int intermediate_size,
                         int num_spec_tokens,
                         cudaStream_t stream);

/**
 * Argmax kernel: find the index of the maximum value along the last dimension.
 * @param logits   [num_tokens, vocab_size] (FP16)
 * @param indices  [num_tokens] (int32)
 * @param num_tokens
 * @param vocab_size
 */
void DFlashArgmaxKernel(const void* logits,
                        void* indices,
                        int num_tokens,
                        int vocab_size,
                        cudaStream_t stream);

/**
 * Softmax + sampling kernel for draft tokens.
 * For each row in logits, compute softmax and sample.
 */
void DFlashSampleKernel(const void* logits,
                        void* indices,
                        const void* random_vals,
                        int num_tokens,
                        int vocab_size,
                        cudaStream_t stream);

}  // namespace turbomind
