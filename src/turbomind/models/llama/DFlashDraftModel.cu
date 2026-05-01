

#include "src/turbomind/models/llama/DFlashDraftModel.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/dflash_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/core/logger.h"

#include <algorithm>
#include <cstring>

namespace turbomind {

// ──────────────────────────────────────────
// CUDA kernel helpers
// ──────────────────────────────────────────

__global__ void DFlashAddResidualKernel(half* dst, const half* src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __hadd(dst[idx], src[idx]);
}

__global__ void DFlashSiluMulKernel(half* gate, const half* up, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float sig = 1.f / (1.f + expf(-g));
        gate[idx] = __float2half(sig * g * __half2float(up[idx]));
    }
}

// Split QKV kernel: extracts Q, K, V from QKV tensor
// QKV: [num_rows, 3*hidden] → Q: [num_rows, hidden], K: [num_rows, hidden], V: [num_rows, hidden]
__global__ void DFlashSplitQKVKernel(
    const half* __restrict__ qkv,   // [num_rows, 3*hidden]
    half* __restrict__ q,           // [num_rows, hidden]
    half* __restrict__ k,           // [num_rows, hidden]
    half* __restrict__ v,           // [num_rows, hidden]
    int num_rows,
    int hidden)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const half* qkv_row = qkv + row * 3 * hidden;
    half* q_row = q + row * hidden;
    half* k_row = k + row * hidden;
    half* v_row = v + row * hidden;

    // Each thread processes multiple elements
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        q_row[i] = qkv_row[i];                // Q starts at offset 0
        k_row[i] = qkv_row[hidden + i];      // K starts at offset hidden
        v_row[i] = qkv_row[2 * hidden + i];  // V starts at offset 2*hidden
    }
}

// ──────────────────────────────────────────
// cuBLAS GEMM helper (FP16)
// C = A * B, both row-major, no transpose
// A: [m, k], B: [k, n] → C: [m, n]
// ──────────────────────────────────────────

static void GemmFP16ComputeFP32(cublasHandle_t cublas,
                                const half* A,
                                const half* B,
                                half* C,
                                int m,
                                int n,
                                int k)
{
    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasGemmEx(cublas,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 B, CUDA_R_16F, n,
                 A, CUDA_R_16F, k,
                 &beta,
                 C, CUDA_R_16F, n,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
}

// ──────────────────────────────────────────
// DFlashDraftModel implementation
// ──────────────────────────────────────────

DFlashDraftModel::DFlashDraftModel(const ModelParam& model,
                                   const EngineParam& engine,
                                   const Context& ctx)
    : hidden_size_(model.hidden_units)
    , num_draft_layers_(8)
    , num_aux_layers_(5)
    , num_spec_tokens_(8)
    , target_layer_ids_{1, 10, 19, 28, 37}
    , vocab_size_(model.vocab_size)
{
    (void)engine;
    (void)ctx;

    weight_ = std::make_unique<DFlashDraftWeight>();

    // Create cuBLAS handle
    cublasCreate(&cublas_);

    TM_LOG_INFO("[DFlash] Draft model created: hidden=%d layers=%d spec=%d vocab=%d",
                hidden_size_, num_draft_layers_, num_spec_tokens_, vocab_size_);
}

DFlashDraftModel::~DFlashDraftModel()
{
    if (cublas_) {
        cublasDestroy(cublas_);
    }
}

void DFlashDraftModel::ExtractAuxHidden(
    const std::vector<Tensor>& layer_outputs,
    std::vector<Tensor>& aux_states)
{
    aux_states.clear();
    aux_states.resize(num_aux_layers_);

    for (int i = 0; i < num_aux_layers_; ++i) {
        int layer_id = target_layer_ids_[i];
        if (layer_id < static_cast<int>(layer_outputs.size())) {
            aux_states[i] = layer_outputs[layer_id];
        }
        else {
            TM_LOG_WARNING("[DFlash] Layer %d not available in layer_outputs", layer_id);
        }
    }

    TM_LOG_DEBUG("[DFlash] Extracted %d aux hidden states", (int)aux_states.size());
}

void DFlashDraftModel::GenerateDraft(
    const std::vector<Tensor>& aux_states,
    Tensor& draft_tokens,
    Tensor& draft_logits)
{
    if (aux_states.empty()) {
        TM_LOG_WARNING("[DFlash] No aux hidden states available");
        return;
    }

    const auto stream = core::Context::stream().handle();
    const int hidden  = hidden_size_;
    const int head_dim = GetDraftWeight()->head_dim;
    const int num_heads = GetDraftWeight()->num_attention_heads;
    const int inter_size = GetDraftWeight()->intermediate_size;

    // Context length: use the token count from first aux state
    const int num_ctx = aux_states[0].shape(0);
    const auto dtype  = aux_states[0].dtype();  // Should be FP16

    // ── 1) Build context hidden from aux states ──
    // Average the 5 aux states: [num_ctx, hidden]
    Tensor ctx_hidden = Tensor{{num_ctx, hidden}, dtype, kDEVICE};
    {
        // Simple average: sum all aux states, then multiply by 1/num_aux
        // For now, use the first aux state as context (can be improved)
        cudaMemcpyAsync(ctx_hidden.raw_data(),
                        const_cast<void*>(aux_states[0].raw_data()),
                        num_ctx * hidden * 2,  // FP16: 2 bytes per element
                        cudaMemcpyDeviceToDevice,
                        stream);
    }

    // ── 2) Compute context K and V ──
    // context_K = ctx_hidden @ W_ctx_k  → [num_ctx, hidden]
    // context_V = ctx_hidden @ W_ctx_v  → [num_ctx, hidden]
    // For simplicity, use draft model's own K/V projection weights
    // but we need separate context K/V projections.
    //
    // For now: derive context K/V from context hidden directly
    // by using the draft model's QKV split weights.
    Tensor ctx_k = Tensor{{num_ctx, hidden}, dtype, kDEVICE};
    Tensor ctx_v = Tensor{{num_ctx, hidden}, dtype, kDEVICE};

    // Split QKV weight into Q, K, V components
    // QKV weight is [hidden, 3*hidden] row-major
    // Q = cols [0, hidden), K = cols [hidden, 2*hidden), V = cols [2*hidden, 3*hidden)
    if (GetDraftWeight()->d_attn_qkv_weight[0]) {
        const int qkv_total = 3 * hidden;
        const half* qkv_w = static_cast<const half*>(GetDraftWeight()->d_attn_qkv_weight[0].raw_data());

        // Extract K weight: rows [0:hidden], cols [hidden:2*hidden]
        // Extract V weight: rows [0:hidden], cols [2*hidden:3*hidden]
        // For efficiency, create strided views or copy the relevant columns

        // Simplified: compute full QKV for context, then split
        Tensor ctx_qkv = Tensor{{num_ctx, qkv_total}, dtype, kDEVICE};
        GemmFP16ComputeFP32(cublas_,
                            static_cast<const half*>(ctx_hidden.raw_data()),
                            qkv_w,
                            static_cast<half*>(ctx_qkv.raw_data()),
                            num_ctx, qkv_total, hidden);

        // Split: Q, K, V each [num_ctx, hidden]
        const half* qkv_data = static_cast<const half*>(ctx_qkv.raw_data());
        half* k_data = static_cast<half*>(ctx_k.raw_data());
        half* v_data = static_cast<half*>(ctx_v.raw_data());

        // 使用 kernel 替代循环 cudaMemcpyAsync
        const int threads = 256;
        DFlashSplitQKVKernel<<<num_ctx, threads, 0, stream>>>(qkv_data, nullptr, k_data, v_data, num_ctx, hidden);
    }
    else {
        // Fallback: zero K/V if no weights loaded
        cudaMemsetAsync(ctx_k.raw_data(), 0, num_ctx * hidden * 2, stream);
        cudaMemsetAsync(ctx_v.raw_data(), 0, num_ctx * hidden * 2, stream);
    }

    // ── 3) Initialize draft hidden states ──
    Tensor draft_hidden = Tensor{{num_spec_tokens_, hidden}, dtype, kDEVICE};
    cudaMemsetAsync(draft_hidden.raw_data(), 0, num_spec_tokens_ * hidden * 2, stream);

    // ── 4) 8 decoder layers ──
    for (int layer = 0; layer < num_draft_layers_; ++layer) {
        const int h = hidden;

        // ── 4a. Input RMSNorm ──
        Tensor norm1 = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};
        invokeRMSNorm(norm1, draft_hidden, GetDraftWeight()->d_input_layernorm[layer],
                      GetDraftWeight()->rms_norm_eps, stream);
        // 移除 sync_check_cuda_error() - 避免同步

        // ── 4b. QKV GEMM: [num_spec, hidden] @ [hidden, 3*hidden] → [num_spec, 3*hidden] ──
        const half* qkv_w = static_cast<const half*>(GetDraftWeight()->d_attn_qkv_weight[layer].raw_data());
        Tensor qkv = Tensor{{num_spec_tokens_, 3 * h}, dtype, kDEVICE};

        if (qkv_w) {
            GemmFP16ComputeFP32(cublas_,
                                static_cast<const half*>(norm1.raw_data()),
                                qkv_w,
                                static_cast<half*>(qkv.raw_data()),
                                num_spec_tokens_, 3 * h, h);
        }

        // ── 4c. Split Q/K/V and reshape for attention ──
        // Q, K, V each [num_spec, hidden] then reshape to [num_spec, num_heads, head_dim]
        const half* qkv_data = static_cast<const half*>(qkv.raw_data());
        Tensor q_flat = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};
        Tensor k_flat = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};
        Tensor v_flat = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};

        // 使用 kernel 替代循环 cudaMemcpyAsync
        DFlashSplitQKVKernel<<<num_spec_tokens_, 256, 0, stream>>>(qkv_data, static_cast<half*>(q_flat.raw_data()), static_cast<half*>(k_flat.raw_data()), static_cast<half*>(v_flat.raw_data()), num_spec_tokens_, h);

        // ── 4d. DFlash Attention ──
        // Query from draft, Key/Value from context + draft
        Tensor attn_out = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};

        DFlashAttentionKernel(
            static_cast<const void*>(q_flat.raw_data()),
            static_cast<const void*>(ctx_k.raw_data()),
            static_cast<const void*>(k_flat.raw_data()),
            static_cast<const void*>(ctx_v.raw_data()),
            static_cast<const void*>(v_flat.raw_data()),
            static_cast<void*>(attn_out.raw_data()),
            num_spec_tokens_,
            num_ctx,
            h,
            head_dim,
            stream);

        // ── 4e. Output projection + residual ──
        const half* o_w = static_cast<const half*>(GetDraftWeight()->d_attn_o_weight[layer].raw_data());
        Tensor attn_proj = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};

        if (o_w) {
            GemmFP16ComputeFP32(cublas_,
                                static_cast<const half*>(attn_out.raw_data()),
                                o_w,
                                static_cast<half*>(attn_proj.raw_data()),
                                num_spec_tokens_, h, h);
        }

        // Add residual: draft_hidden = draft_hidden + attn_proj
        int threads = 256 < (num_spec_tokens_ * h) ? 256 : (num_spec_tokens_ * h);
        DFlashAddResidualKernel<<<(num_spec_tokens_ * h + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<half*>(draft_hidden.raw_data()),
            static_cast<const half*>(attn_proj.raw_data()),
            num_spec_tokens_ * h);

        // ── 4f. Post-attention RMSNorm ──
        Tensor norm2 = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};
        invokeRMSNorm(norm2, draft_hidden, GetDraftWeight()->d_post_layernorm[layer],
                      GetDraftWeight()->rms_norm_eps, stream);
        // 移除 sync_check_cuda_error() - 避免同步

        // ── 4g. MLP ──
        // gate_up = silu(x @ Wg) * (x @ Wu), out = gate_up @ Wd
        // Split gate_up_proj into gate and up projections
        const half* gu_w = static_cast<const half*>(GetDraftWeight()->d_gate_up_proj[layer].raw_data());
        const half* d_w  = static_cast<const half*>(GetDraftWeight()->d_down_proj[layer].raw_data());

        if (gu_w && d_w) {
            // gate_up_proj is [hidden, 2*inter_size] — split in half
            Tensor gate_out  = Tensor{{num_spec_tokens_, inter_size}, dtype, kDEVICE};
            Tensor up_out    = Tensor{{num_spec_tokens_, inter_size}, dtype, kDEVICE};
            Tensor gate_proj = Tensor{{num_spec_tokens_, inter_size}, dtype, kDEVICE};

            // GEMM: norm2 @ gate_proj_weight → gate_out
            // gate_up weight is [hidden, 2*inter], gate is first [hidden, inter], up is [hidden, inter]
            GemmFP16ComputeFP32(cublas_,
                                static_cast<const half*>(norm2.raw_data()),
                                gu_w,
                                static_cast<half*>(gate_out.raw_data()),
                                num_spec_tokens_, inter_size, h);

            // GEMM: norm2 @ up_proj_weight → up_out (second half of gate_up weight)
            // up weight starts at offset hidden * inter_size in column-major layout
            GemmFP16ComputeFP32(cublas_,
                                static_cast<const half*>(norm2.raw_data()),
                                gu_w + hidden * inter_size,
                                static_cast<half*>(up_out.raw_data()),
                                num_spec_tokens_, inter_size, h);

            // silu(gate) * up
            int threads_m = 256 < (num_spec_tokens_ * inter_size) ? 256 : (num_spec_tokens_ * inter_size);
            DFlashSiluMulKernel<<<(num_spec_tokens_ * inter_size + threads_m - 1) / threads_m, threads_m, 0, stream>>>(
                static_cast<half*>(gate_out.raw_data()),
                static_cast<const half*>(up_out.raw_data()),
                num_spec_tokens_ * inter_size);

            // GEMM: silu_result @ down_proj → draft_hidden (for residual)
            Tensor mlp_out = Tensor{{num_spec_tokens_, h}, dtype, kDEVICE};
            GemmFP16ComputeFP32(cublas_,
                                static_cast<const half*>(gate_out.raw_data()),
                                d_w,
                                static_cast<half*>(mlp_out.raw_data()),
                                num_spec_tokens_, h, inter_size);

            // Add residual
            DFlashAddResidualKernel<<<(num_spec_tokens_ * h + threads - 1) / threads, threads, 0, stream>>>(
                static_cast<half*>(draft_hidden.raw_data()),
                static_cast<const half*>(mlp_out.raw_data()),
                num_spec_tokens_ * h);
        }
    }

    // ── 5) Final norm ──
    Tensor final_norm = Tensor{{num_spec_tokens_, hidden}, dtype, kDEVICE};
    invokeRMSNorm(final_norm, draft_hidden, GetDraftWeight()->d_input_layernorm[0],  // Use first layer norm as fallback
                  GetDraftWeight()->rms_norm_eps, stream);
    // 移除 sync_check_cuda_error() - 避免同步

    // ── 6) LM head projection → logits ──
    // [num_spec, hidden] @ [hidden, vocab] → [num_spec, vocab]
    const half* lm_w = static_cast<const half*>(GetDraftWeight()->lm_head.raw_data());
    if (lm_w && GetDraftWeight()->lm_head) {
        draft_logits = Tensor{{num_spec_tokens_, vocab_size_}, dtype, kDEVICE};
        GemmFP16ComputeFP32(cublas_,
                            static_cast<const half*>(final_norm.raw_data()),
                            lm_w,
                            static_cast<half*>(draft_logits.raw_data()),
                            num_spec_tokens_, vocab_size_, hidden);
    }

    // ── 7) Argmax → draft tokens ──
    if (draft_logits) {
        draft_tokens = Tensor{{num_spec_tokens_}, kInt32, kDEVICE};
        DFlashArgmaxKernel(static_cast<const void*>(draft_logits.raw_data()),
                           draft_tokens.raw_data(),
                           num_spec_tokens_,
                           vocab_size_,
                           stream);
    }

    TM_LOG_DEBUG("[DFlash] Generated %d draft tokens", num_spec_tokens_);
}

// ──────────────────────────────────────────
// GPU-based speculative verification
//
// For each spec position:
// 1. Find argmax of target logits → target_token
// 2. If target_token == draft_token → accept
// 3. Else → reject (use target_token as accepted token)
//
// All GPU-side, no CPU-GPU sync needed.
// ──────────────────────────────────────────

__global__ void DFlashVerifyDraftKernel(
    const int* __restrict__ draft_tokens,     // [num_spec]
    const half* __restrict__ target_logits,   // [num_spec, vocab_size]
    int* __restrict__ accepted_tokens,        // [num_spec]
    int* __restrict__ accept_mask,            // [num_spec]
    int num_spec,
    int vocab_size)
{
    const int i = blockIdx.x;  // spec position index
    if (i >= num_spec) return;

    // ── 1) Argmax of target logits ──
    const half* logits_row = target_logits + i * vocab_size;
    float max_val = -1e20f;
    int max_idx = 0;

    // Use shared memory for block-level reduction
    extern __shared__ char smem_char[];
    float* s_max = reinterpret_cast<float*>(smem_char);
    int*   s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

    float local_max = -1e20f;
    int local_idx = 0;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float v_float = __half2float(logits_row[v]);
        if (v_float > local_max) {
            local_max = v_float;
            local_idx = v;
        }
    }

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_idx;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_max[threadIdx.x] < s_max[threadIdx.x + stride]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    int target_token = s_idx[0];

    // ── 2) Compare with draft token ──
    int draft = draft_tokens[i];
    if (draft == target_token) {
        accepted_tokens[i] = draft;
        accept_mask[i] = 1;
    } else {
        accepted_tokens[i] = target_token;
        accept_mask[i] = 0;
    }
}

// ──────────────────────────────────────────
// Count accepted tokens kernel
// Simple reduction to count how many tokens were accepted
// ──────────────────────────────────────────

__global__ void DFlashCountAcceptedKernel(
    const int* __restrict__ accept_mask,  // [num_spec]
    int* __restrict__ count,              // [1]
    int num_spec)
{
    extern __shared__ char smem_char[];
    int* s_count = reinterpret_cast<int*>(smem_char);

    int local_count = 0;
    for (int i = threadIdx.x; i < num_spec; i += blockDim.x) {
        local_count += accept_mask[i];
    }

    s_count[threadIdx.x] = local_count;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_count[threadIdx.x] += s_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        count[0] = s_count[0];
    }
}

void DFlashVerifyDraftGPU(
    const Tensor& draft_tokens,
    const Tensor& target_logits,
    Tensor& accepted_tokens,
    Tensor& accept_mask,
    int num_spec_tokens)
{
    const int num_spec = num_spec_tokens;
    const int vocab_size = target_logits.shape(1);
    const auto stream = core::Context::stream().handle();

    accepted_tokens = Tensor{{num_spec}, kInt32, kDEVICE};
    accept_mask = Tensor{{num_spec}, kInt32, kDEVICE};

    // Launch verification kernel
    // One block per spec position (8 blocks for 8 spec tokens)
    // Each block handles vocab_size with threads
    int threads = std::min(256, vocab_size);
    int smem = threads * (sizeof(float) + sizeof(int));

    const half* logits_data = static_cast<const half*>(target_logits.raw_data());

    DFlashVerifyDraftKernel<<<num_spec, threads, smem, stream>>>(
        static_cast<const int*>(draft_tokens.raw_data()),
        logits_data,
        static_cast<int*>(accepted_tokens.raw_data()),
        static_cast<int*>(accept_mask.raw_data()),
        num_spec,
        vocab_size);

    // Count accepted tokens for logging
    Tensor d_count = Tensor{{1}, kInt32, kDEVICE};
    Tensor h_count = Tensor{{1}, kInt32, kCPUpinned};

    DFlashCountAcceptedKernel<<<1, std::min(256, num_spec), 256 * sizeof(int), stream>>>(
        static_cast<const int*>(accept_mask.raw_data()),
        static_cast<int*>(d_count.raw_data()),
        num_spec);

    // Copy count asynchronously (no sync - let it complete in background)
    cudaMemcpyAsync(static_cast<int*>(h_count.raw_data()),
                    static_cast<const int*>(d_count.raw_data()),
                    sizeof(int), cudaMemcpyDeviceToDevice,
                    stream);

    // Note: We intentionally do NOT sync here.
    // The count is only for logging and can be read later.
    // If we need the count immediately, we can sync after the log.
    // For now, use a rough estimate.
    TM_LOG_DEBUG("[DFlash] Verification launched on GPU for %d spec tokens", num_spec);
}

void DFlashDraftModel::VerifyDraft(
    const Tensor& draft_tokens,
    const Tensor& target_logits,
    Tensor& accepted_tokens,
    Tensor& accept_mask)
{
    const int num_spec = num_spec_tokens_;
    int vocab_size = target_logits.shape(1);

    accepted_tokens = Tensor{{num_spec}, kInt32, kDEVICE};
    accept_mask = Tensor{{num_spec}, kInt32, kDEVICE};

    // Check dtype of target_logits
    if (target_logits.dtype() == kFloat) {
        // Convert to FP16 first
        const auto stream = core::Context::stream().handle();
        Tensor logits_fp16 = Tensor{{num_spec, vocab_size}, kFloat16, kDEVICE};
        invokeCastFloat2D(target_logits, logits_fp16, stream);
        DFlashVerifyDraftGPU(draft_tokens, logits_fp16, accepted_tokens, accept_mask, num_spec);
    }
    else {
        DFlashVerifyDraftGPU(draft_tokens, target_logits, accepted_tokens, accept_mask, num_spec);
    }

    // Note: No sync here. The verification is fully asynchronous.
    // The caller will sync when it needs the results.
}

}  // namespace turbomind
