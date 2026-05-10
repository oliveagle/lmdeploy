

#include <cuda_fp16.h>

#include "src/turbomind/models/llama/dflash_kernels.h"
#include "src/turbomind/core/logger.h"

namespace turbomind {

// ──────────────────────────────────────────
// DFlash Attention Kernel (FP16)
//
// One CTA per (spec_pos, head). Each thread
// processes a slice of head_dim.
//
// Grid:  (num_draft, num_heads)
// Block:  128 threads
// ──────────────────────────────────────────

template <int THREADS>
__global__ void DFlashAttnKernelFP16(
    const half* __restrict__ query,       // [num_draft, num_heads, head_dim]
    const half* __restrict__ key_ctx,     // [num_ctx, num_heads, head_dim]
    const half* __restrict__ key_draft,   // [num_draft, num_heads, head_dim]
    const half* __restrict__ val_ctx,     // [num_ctx, num_heads, head_dim]
    const half* __restrict__ val_draft,   // [num_draft, num_heads, head_dim]
    half* __restrict__ output,            // [num_draft, num_heads, head_dim]
    int num_ctx,
    int num_draft,
    int head_dim,
    float scale)
{
    const int s = blockIdx.x;   // draft position index
    const int h = blockIdx.y;   // head index
    if (s >= num_draft) return;

    const int tid     = threadIdx.x;
    const int total   = num_ctx + num_draft;  // Non-causal: all context + all draft tokens

    // Offset into per-head data (row-major layout)
    const half* q     = query + s * blockDim.y * head_dim + h * head_dim;
    const half* kc    = key_ctx + h * head_dim;
    const half* kd    = key_draft + s * blockDim.y * head_dim + h * head_dim;
    const half* vc    = val_ctx + h * head_dim;
    const half* vd    = val_draft + s * blockDim.y * head_dim + h * head_dim;
    half* out         = output + s * blockDim.y * head_dim + h * head_dim;

    // Shared memory layout:
    //   logits[total]     – attention scores (FP32 for numerical stability)
    //   reduce_max[THREADS]
    //   reduce_sum[THREADS]
    extern __shared__ char smem_char[];
    float* logits        = reinterpret_cast<float*>(smem_char);
    float* rmax          = logits + total;
    float* rsum          = rmax + THREADS;

    // ── 1) Q·K dot products ──────────────────
    float lmax = -1e20f;

    // context keys
    for (int i = tid; i < num_ctx; i += THREADS) {
        float dot = 0.f;
#pragma unroll
        for (int d = 0; d < 128; ++d) {
            if (d < head_dim)
                dot += __half2float(q[d]) * __half2float(kc[i * head_dim + d]);
        }
        dot *= scale;
        logits[i] = dot;
        lmax = fmaxf(lmax, dot);
    }

    // draft keys (NON-CAUSAL: all draft tokens can see all draft tokens)
    // This matches lucebox-hub/dflash behavior for better draft quality
    const int ds = num_ctx;
    for (int i = tid; i < num_draft; i += THREADS) {
        float dot = 0.f;
#pragma unroll
        for (int d = 0; d < 128; ++d) {
            if (d < head_dim)
                dot += __half2float(q[d]) * __half2float(kd[i * head_dim + d]);
        }
        dot *= scale;
        logits[ds + i] = dot;
        lmax = fmaxf(lmax, dot);
    }

    // ── 2) max reduction ─────────────────────
    rmax[tid] = lmax;
    __syncthreads();
    for (int s2 = THREADS >> 1; s2; s2 >>= 1) {
        if (tid < s2) rmax[tid] = fmaxf(rmax[tid], rmax[tid + s2]);
        __syncthreads();
    }
    float gmax = rmax[0];
    __syncthreads();

    // ── 3) softmax sum ───────────────────────
    float lsum = 0.f;
    for (int i = tid; i < total; i += THREADS) {
        float e = expf(logits[i] - gmax);
        logits[i] = e;
        lsum += e;
    }
    rsum[tid] = lsum;
    __syncthreads();
    for (int s2 = THREADS >> 1; s2; s2 >>= 1) {
        if (tid < s2) rsum[tid] += rsum[tid + s2];
        __syncthreads();
    }
    float gsum = rsum[0] + 1e-8f;
    __syncthreads();

    // ── 4) weighted value sum (master thread) ─
    if (tid == 0) {
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.f;
            for (int i = 0; i < num_ctx; ++i)
                acc += logits[i] * __half2float(vc[i * head_dim + d]);
            for (int i = 0; i < num_draft; ++i)
                acc += logits[ds + i] * __half2float(vd[i * head_dim + d]);
            out[d] = __float2half(acc / gsum);
        }
    }
}

// Host wrapper
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
                           cudaStream_t stream)
{
    int num_heads = hidden_size / head_dim;
    int smem = (num_ctx + num_draft) * sizeof(float) + 2 * 128 * sizeof(float);
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    DFlashAttnKernelFP16<128><<<dim3(num_draft, num_heads), 128, smem, stream>>>(
        static_cast<const half*>(query),
        static_cast<const half*>(key_ctx),
        static_cast<const half*>(key_draft),
        static_cast<const half*>(value_ctx),
        static_cast<const half*>(value_draft),
        static_cast<half*>(output),
        num_ctx, num_draft, head_dim, scale);
}


// ──────────────────────────────────────────
// DFlash MLP Kernel (FP16)
//
// gate_up = silu(x @ Wg) * (x @ Wu)
// output  = gate_up @ Wd
//
// Each block handles one draft position.
// Uses FP32 accumulation for numerical stability.
// ──────────────────────────────────────────

__global__ void DFlashMLPFP16(
    const half* __restrict__ x,           // [num_draft, hidden_size] (row-major)
    const half* __restrict__ gate_proj,   // [hidden_size, inter_size] (row-major)
    const half* __restrict__ up_proj,     // [hidden_size, inter_size] (row-major)
    const half* __restrict__ down_proj,   // [inter_size, hidden_size] (row-major)
    half* __restrict__ out,               // [num_draft, hidden_size] (row-major)
    int hidden_size,
    int inter_size)
{
    const int s = blockIdx.x;
    if (s >= gridDim.x) return;

    const half* xs = x + s * hidden_size;
    half*      os  = out + s * hidden_size;

    extern __shared__ half sm_gate[];  // [inter_size]

    // ── gate + up projection ────────────────
    for (int i = threadIdx.x; i < inter_size; i += blockDim.x) {
        float g = 0.f, u = 0.f;
        for (int j = 0; j < hidden_size; ++j) {
            g += __half2float(xs[j]) * __half2float(gate_proj[j * inter_size + i]);
            u += __half2float(xs[j]) * __half2float(up_proj[j * inter_size + i]);
        }
        // silu(g) * u
        float sig = 1.f / (1.f + expf(-g));
        sm_gate[i] = __float2half(sig * g * u);
    }
    __syncthreads();

    // ── down projection ─────────────────────
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < inter_size; ++j)
            acc += __half2float(sm_gate[j]) * __half2float(down_proj[j * hidden_size + i]);
        os[i] = __float2half(acc);
    }
}

void DFlashMLPKernel(const void* hidden,
                     const void* gate,
                     const void* up,
                     const void* down,
                     void* output,
                     int num_draft,
                     int hidden_size,
                     int intermediate_size,
                     cudaStream_t stream)
{
    int threads = min(256, intermediate_size);
    int smem    = intermediate_size * sizeof(half);
    DFlashMLPFP16<<<num_draft, threads, smem, stream>>>(
        static_cast<const half*>(hidden),
        static_cast<const half*>(gate),
        static_cast<const half*>(up),
        static_cast<const half*>(down),
        static_cast<half*>(output),
        hidden_size, intermediate_size);
}


// ──────────────────────────────────────────
// DFlash Argmax Kernel (FP16)
//
// Find index of max value along last dimension.
// Each block processes one row (one token position).
// ──────────────────────────────────────────

__global__ void DFlashArgmaxFP16(
    const half* __restrict__ logits,  // [num_tokens, vocab_size]
    int* __restrict__ indices,        // [num_tokens]
    int vocab_size)
{
    const int t = blockIdx.x;  // token index
    if (t >= gridDim.x) return;

    const half* row = logits + t * vocab_size;
    float max_val = -1e20f;
    int max_idx = 0;

    // Each thread handles a chunk of vocab
    float local_max = -1e20f;
    int local_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = __half2float(row[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Block-level reduction using shared memory
    extern __shared__ char smem_char[];
    float* s_max = reinterpret_cast<float*>(smem_char);
    int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_max[threadIdx.x] < s_max[threadIdx.x + stride]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        indices[t] = s_idx[0];
    }
}

void DFlashArgmaxKernel(const void* logits,
                        void* indices,
                        int num_tokens,
                        int vocab_size,
                        cudaStream_t stream)
{
    int threads = min(256, vocab_size);
    int smem = 256 * (sizeof(float) + sizeof(int));
    DFlashArgmaxFP16<<<num_tokens, threads, smem, stream>>>(
        static_cast<const half*>(logits),
        static_cast<int*>(indices),
        vocab_size);
}


// ──────────────────────────────────────────
// DFlash Forward Kernel (stub - delegates to per-layer kernels)
// ──────────────────────────────────────────

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
                         cudaStream_t stream)
{
    // The full forward pass is too complex for a single kernel
    // with large hidden_size. Instead, DFlashDraftModel::GenerateDraft
    // calls individual attention + MLP + norm kernels layer by layer.
    (void)aux_hidden; (void)qkv_proj; (void)o_proj;
    (void)gate_up_proj; (void)down_proj; (void)input_ln;
    (void)post_ln; (void)output; (void)num_layers;
    (void)hidden_size; (void)intermediate_size;
    (void)num_spec_tokens; (void)stream;
}


// ──────────────────────────────────────────
// DFlash Sample Kernel (FP16)
//
// Softmax + sampling: for each row in logits,
// compute softmax and sample based on random_val.
// ──────────────────────────────────────────

__global__ void DFlashSampleFP16(
    const half* __restrict__ logits,  // [num_tokens, vocab_size]
    const float* __restrict__ rand,   // [num_tokens]
    int* __restrict__ indices,        // [num_tokens]
    int vocab_size)
{
    const int t = blockIdx.x;
    if (t >= gridDim.x) return;

    const half* row = logits + t * vocab_size;

    // Find max for numerical stability
    float local_max = -1e20f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(row[i]));
    }

    // Block-level max reduction
    extern __shared__ char smem_char[];
    float* s_max = reinterpret_cast<float*>(smem_char);
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];

    // Compute softmax + sum
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = expf(__half2float(row[i]) - max_val);
        local_sum += v;
    }

    // Block-level sum reduction
    float* s_sum = s_max;  // reuse memory
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_val = s_sum[0] + 1e-8f;

    // Sample
    float target = rand[t] * sum_val;
    float cumulative = 0.f;
    int idx = 0;

    for (int i = 0; i < vocab_size; ++i) {
        cumulative += expf(__half2float(row[i]) - max_val) / sum_val;
        if (cumulative >= target) {
            idx = i;
            break;
        }
    }

    indices[t] = idx;
}

void DFlashSampleKernel(const void* logits,
                        void* indices,
                        const void* random_vals,
                        int num_tokens,
                        int vocab_size,
                        cudaStream_t stream)
{
    int threads = min(256, vocab_size);
    int smem = 256 * sizeof(float);
    DFlashSampleFP16<<<num_tokens, threads, smem, stream>>>(
        static_cast<const half*>(logits),
        static_cast<const float*>(random_vals),
        static_cast<int*>(indices),
        vocab_size);
}

}  // namespace turbomind
