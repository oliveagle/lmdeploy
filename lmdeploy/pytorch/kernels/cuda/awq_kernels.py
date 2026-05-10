# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import triton
from triton import language as tl

# 设置 Triton benchmark 缓存大小以避免 OOM
# 默认值是总显存大小，在 16GB V100 上会分配 4GB 作为 cache
os.environ['TRITON_BENCHMARK_CACHE_SIZE_KB'] = '262144'  # 256MB instead of 4GB

def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_N': 64,
            'GROUP_SIZE_M': 8,
        }, num_stages=3, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_N': 128,
            'GROUP_SIZE_M': 8,
        }, num_stages=3, num_warps=4),
    ]


@triton.jit
def _dequant_s4_to_f16x2(weight, shift: tl.constexpr, is_top: tl.constexpr):

    immLut: tl.constexpr = (0xf0 & 0xcc) | 0xaa
    BOTTOM_MASK: tl.constexpr = 0x000f000f
    TOP_MASK: tl.constexpr = 0x00f000f0
    I4s_TO_F16s_MAGIC_NUM: tl.constexpr = 0x64006400
    FP16_TOP_MAGIC_NUM: tl.constexpr = 0x64006400
    ONE_SIXTEENTH: tl.constexpr = 0x2c002c00
    NEG_64: tl.constexpr = 0xd400d400

    if shift:
        weight = weight >> 8

    if is_top:
        return tl.inline_asm_elementwise("""{
        .reg .b32 tmp;
        lop3.b32 tmp, $2, $3, $4, $5;
        fma.rn.f16x2 tmp, tmp, $6, $7;
        mov.b32 {$0, $1}, tmp;
    }""",
                                         '=h,=h,r,n,n,n,r,r',
                                         args=[weight, TOP_MASK, I4s_TO_F16s_MAGIC_NUM, immLut, ONE_SIXTEENTH, NEG_64],
                                         dtype=(tl.float16, tl.float16),
                                         is_pure=True,
                                         pack=1)
    else:
        return tl.inline_asm_elementwise("""{
        .reg .b32 tmp;
        lop3.b32 tmp, $2, $3, $4, $5;
        sub.f16x2 tmp, tmp, $6;
        mov.b32 {$0, $1}, tmp;
    }""",
                                         '=h,=h,r,n,n,n,r',
                                         args=[weight, BOTTOM_MASK, I4s_TO_F16s_MAGIC_NUM, immLut, FP16_TOP_MAGIC_NUM],
                                         dtype=(tl.float16, tl.float16),
                                         is_pure=True,
                                         pack=1)


@triton.jit
def _unpack_weight(weight):
    """Unpack weight."""
    # broadcast and shift
    width: tl.constexpr = 8
    BLOCK_SIZE_K: tl.constexpr = weight.shape[0]
    BLOCK_SIZE_QN: tl.constexpr = weight.shape[1]
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_QN * width

    w0, w1 = _dequant_s4_to_f16x2(weight, False, False)
    w2, w3 = _dequant_s4_to_f16x2(weight, False, True)
    w4, w5 = _dequant_s4_to_f16x2(weight, True, False)
    w6, w7 = _dequant_s4_to_f16x2(weight, True, True)

    w04 = tl.join(w0, w4)
    w15 = tl.join(w1, w5)
    w26 = tl.join(w2, w6)
    w37 = tl.join(w3, w7)
    w0246 = tl.join(w04, w26)
    w1357 = tl.join(w15, w37)
    weight = tl.join(w0246, w1357)

    return weight.reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)


@triton.jit
def awq_linear_kernel(
        a_ptr,
        qw_ptr,
        s_ptr,
        qz_ptr,
        c_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am,
        stride_ak: tl.constexpr,  #
        stride_wk: tl.constexpr,
        stride_wn: tl.constexpr,  #
        stride_sk: tl.constexpr,
        stride_sn: tl.constexpr,  #
        stride_zk: tl.constexpr,
        stride_zn: tl.constexpr,  #
        stride_cm,
        stride_cn: tl.constexpr,
        # Meta-parameters
        SPLIT_K: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.

    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    kid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    BLOCK_SIZE_QN: tl.constexpr = BLOCK_SIZE_N // 8
    offs_wn = pid_n * BLOCK_SIZE_QN + tl.arange(0, BLOCK_SIZE_QN)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    qw_ptrs = qw_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)
    s_ptrs = s_ptr + offs_bn * stride_sn
    qz_ptrs = qz_ptr + offs_wn * stride_zn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k_start = kid
    k_last = K // BLOCK_SIZE_K

    # prefetch
    a_ptrs += k_start * BLOCK_SIZE_K * stride_ak
    qw_ptrs += k_start * BLOCK_SIZE_K * stride_wk
    s_ptrs += k_start * stride_sk
    qz_ptrs += k_start * stride_zk
    qw = tl.load(qw_ptrs)
    qz = tl.load(qz_ptrs)[None, :]
    s = tl.load(s_ptrs)[None, :]
    qw_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_wk
    s_ptrs += SPLIT_K * stride_sk
    qz_ptrs += SPLIT_K * stride_zk

    for k in tl.range(k_start, k_last, SPLIT_K, num_stages=NUM_STAGES):

        # unpack b
        z = _unpack_weight(qz)
        w = _unpack_weight(qw)
        b = (w - z) * s

        # load a
        a = tl.load(a_ptrs)

        # load next q
        mask = k + SPLIT_K < k_last
        qz = tl.load(qz_ptrs, mask=mask)[None, :]
        s = tl.load(s_ptrs, mask=mask)[None, :]
        qw = tl.load(qw_ptrs, mask=mask)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_ak
        qw_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_wk
        s_ptrs += SPLIT_K * stride_sk
        qz_ptrs += SPLIT_K * stride_zk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, c, mask=c_mask, sem='relaxed', scope='gpu')
    else:
        tl.store(c_ptrs, c, mask=c_mask)


def awq_linear(x, qweight, scales, qzeros):
    """Awq linear."""
    M = x.size(0)
    K = qweight.size(0)
    N = scales.size(1)
    group_size = K // scales.size(0)
    SPLIT_K = max(1, K // 4096)

    def grid(META):
        """grid."""
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            SPLIT_K,
        )

    if SPLIT_K > 1:
        out = scales.new_zeros(M, N)
    else:
        out = scales.new_empty(M, N)

    props = torch.cuda.get_device_properties(x.device)
    if props.major == 9:
        num_stages = 2
    elif props.major == 8 and props.minor in [6, 9]:
        num_stages = 2
    else:
        num_stages = 3

    BLOCK_SIZE_M = triton.next_power_of_2(M)
    BLOCK_SIZE_M = max(16, min(128, BLOCK_SIZE_M))

    # 直接调用 kernel，不使用 autotuner
    # 使用 BLOCK_SIZE_N = 64，这在 V100 上应该能工作
    # 不使用 triton.autotune 来避免 OOM
    BLOCK_SIZE_N = 64
    GROUP_SIZE_M = 8

    # 构造 grid 和调用
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        SPLIT_K,
    )

    # 调用 kernel 而不是 autotuner
    awq_linear_kernel[grid](
        x,
        qweight,
        scales,
        qzeros,
        out,
        M,
        N,
        K,
        stride_am=x.stride(0),
        stride_ak=x.stride(1),
        stride_wk=qweight.stride(0),
        stride_wn=qweight.stride(1),
        stride_sk=scales.stride(0),
        stride_sn=scales.stride(1),
        stride_zk=qzeros.stride(0),
        stride_zn=qzeros.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        SPLIT_K=SPLIT_K,
        NUM_STAGES=num_stages,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=group_size,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return out


def awq_dequant_weights(qweight, scales, qzeros, w_bit=4, group_size=128):
    """Dequantize AWQ weights from int4 to float16.

    This is a fallback implementation using PyTorch operations.
    Not optimal for performance, but works for correctness.

    Args:
        qweight: Packed quantized weights, shape (in_features, quant_out_features)
                 or (num_experts, in_features, quant_out_features)
        scales: Scaling factors, shape (grouped_in_feats, out_features)
                or (num_experts, grouped_in_feats, out_features)
        qzeros: Packed zeros, shape (grouped_in_feats, quant_out_features)
                or (num_experts, grouped_in_feats, quant_out_features)
        w_bit: Bit width (default 4)
        group_size: Group size for quantization

    Returns:
        Dequantized weights in float16, shape (in_features, out_features)
                or (num_experts, out_features, in_features)
    """
    import torch

    assert w_bit == 4, f"Only w_bit=4 is supported, got {w_bit}"
    elem_per_int = 32 // w_bit

    # Check if we have multiple experts
    has_experts = qweight.dim() == 3
    if has_experts:
        num_experts = qweight.size(0)
        results = []
        for e in range(num_experts):
            expert_qw = qweight[e]
            expert_s = scales[e]
            expert_qz = qzeros[e]
            result = awq_dequant_weights_single_expert(
                expert_qw, expert_s, expert_qz, w_bit, group_size
            )
            results.append(result.t())  # Transpose to (out_features, in_features)
        return torch.stack(results, dim=0)
    else:
        return awq_dequant_weights_single_expert(qweight, scales, qzeros, w_bit, group_size)


def awq_dequant_weights_single_expert(qweight, scales, qzeros, w_bit=4, group_size=128):
    """Dequantize AWQ weights from int4 to float16 for a single expert.

    Args:
        qweight: Packed quantized weights, shape (in_features, quant_out_features)
        scales: Scaling factors, shape (grouped_in_feats, out_features)
        qzeros: Packed zeros, shape (grouped_in_feats, quant_out_features)
        w_bit: Bit width (default 4)
        group_size: Group size for quantization

    Returns:
        Dequantized weights in float16, shape (in_features, out_features)
    """
    import torch

    assert w_bit == 4, f"Only w_bit=4 is supported, got {w_bit}"
    elem_per_int = 32 // w_bit

    in_features, quant_out_features = qweight.shape
    grouped_in_feats, _ = qzeros.shape
    out_features = scales.size(1)

    # Unpack qweight: (in_features, quant_out_features) -> (in_features, out_features)
    qw_unpacked = torch.zeros(in_features, out_features, dtype=torch.int32, device=qweight.device)
    for i in range(elem_per_int):
        shift = i * w_bit
        qw_unpacked[:, i::elem_per_int] = (qweight >> shift) & 0xF

    # Unpack qzeros: (grouped_in_feats, quant_out_features) -> (grouped_in_feats, out_features)
    qz_unpacked = torch.zeros(grouped_in_feats, out_features, dtype=torch.int32, device=qzeros.device)
    for i in range(elem_per_int):
        shift = i * w_bit
        qz_unpacked[:, i::elem_per_int] = (qzeros >> shift) & 0xF

    # Dequantize: (weight - zeros) * scales
    # We need to reshape for broadcasting
    scales_broadcast = scales.reshape(grouped_in_feats, 1, out_features)
    qz_broadcast = qz_unpacked.reshape(grouped_in_feats, 1, out_features)
    qw_reshaped = qw_unpacked.reshape(grouped_in_feats, group_size, out_features)

    weights = (qw_reshaped.to(torch.float16) - qz_broadcast.to(torch.float16)) * scales_broadcast.to(torch.float16)
    weights = weights.reshape(in_features, out_features)

    return weights
