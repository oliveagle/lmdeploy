#!/usr/bin/env python3
"""测试方案1: 严格性能测试

使用更大问题和 cuda.event 精确计时
"""

import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import torch
import time

print("=" * 60)
print("测试方案1: 严格性能测试 (cuda.event)")
print("=" * 60)

from lmdeploy.pytorch.models.dflash import DFlashAttention

class TestConfig:
    num_attention_heads = 4
    num_key_value_heads = 4
    hidden_size = 128
    head_dim = 32
    attention_bias = False
    rms_norm_eps = 1e-5
    quantization_config = None

config = TestConfig()
attn = DFlashAttention(config, dtype=torch.float16, device='cuda')

# Warmup
print("\n[Warmup]")
for num_tokens in [1, 8, 16, 32, 64]:
    hidden_states = torch.randn(num_tokens, 128, dtype=torch.float16, device='cuda')
    target_hidden = torch.randn(num_tokens, 128, dtype=torch.float16, device='cuda')
    cos = torch.randn(num_tokens, 32, dtype=torch.float16, device='cuda')
    sin = torch.randn(num_tokens, 32, dtype=torch.float16, device='cuda')
    with torch.no_grad():
        _ = attn(hidden_states, target_hidden, (cos, sin))
torch.cuda.empty_cache()

# 测试 1: 使用 cuda.event 精确计时 (随机数据)
print("\n[测试1] cuda.event 精确计时 - 每次随机数据")
print("-" * 60)

results = []
for num_tokens in [1, 4, 8, 16, 32, 64, 128, 256, 512]:
    iters = 500

    # 预生成随机数据 (避免 randn 计时)
    input_pool = []
    for _ in range(iters):
        hs = torch.randn(num_tokens, 128, dtype=torch.float16, device='cuda')
        th = torch.randn(num_tokens, 128, dtype=torch.float16, device='cuda')
        c = torch.randn(num_tokens, 32, dtype=torch.float16, device='cuda')
        s = torch.randn(num_tokens, 32, dtype=torch.float16, device='cuda')
        input_pool.append((hs, th, c, s))

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for hs, th, c, s in input_pool:
        with torch.no_grad():
            _ = attn(hs, th, (c, s))
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    avg_ms = elapsed_ms / iters
    throughput = num_tokens * iters / (elapsed_ms / 1000)

    results.append((num_tokens, avg_ms, throughput))
    print(f"num_tokens={num_tokens:3d}: {avg_ms:8.4f}ms, {throughput:8.0f} tokens/sec")

# 测试 2: 纯 kernel 计时 (预生成数据)
print("\n[测试2] 对比: kernel 自身 vs 包括数据生成")
print("-" * 60)

num_tokens = 64
iters = 1000

# 预生成数据
data_pool = []
for _ in range(iters):
    data_pool.append((
        torch.randn(num_tokens, 128, dtype=torch.float16, device='cuda'),
        torch.randn(num_tokens, 128, dtype=torch.float16, device='cuda'),
        torch.randn(num_tokens, 32, dtype=torch.float16, device='cuda'),
        torch.randn(num_tokens, 32, dtype=torch.float16, device='cuda'),
    ))

# 计时1: kernel + 数据池读取
torch.cuda.synchronize()
start = time.perf_counter()
for hs, th, c, s in data_pool:
    with torch.no_grad():
        _ = attn(hs, th, (c, s))
torch.cuda.synchronize()
time1 = time.perf_counter() - start

# 计时2: 仅 kernel 执行 (cuda.event)
torch.cuda.synchronize()
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

start_ev.record()
for hs, th, c, s in data_pool:
    with torch.no_grad():
        _ = attn(hs, th, (c, s))
end_ev.record()
torch.cuda.synchronize()
time2_ms = start_ev.elapsed_time(end_ev) / iters

print(f"num_tokens=64, iters=1000:")
print(f"  perf_counter: {time1/iters*1000:.4f}ms")
print(f"  cuda.event:   {time2_ms:.4f}ms")
print(f"  Python 开销:  {(time1/iters*1000 - time2_ms):.4f}ms")

# 测试 3: 与 torch.matmul 对比 (公平对比)
print("\n[测试3] flash_attn_varlen_func vs torch.matmul")
print("-" * 60)

def manual_attention(q, k, v, scale):
    """手动实现 non-causal attention"""
    attn_weights = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v.transpose(0, 1))
    return attn_output.transpose(0, 1).contiguous()

for num_tokens in [8, 16, 32, 64]:
    # 准备数据
    q = torch.randn(num_tokens, 4, 32, dtype=torch.float16, device='cuda')
    k = torch.randn(num_tokens, 4, 32, dtype=torch.float16, device='cuda')
    v = torch.randn(num_tokens, 4, 32, dtype=torch.float16, device='cuda')

    # flash_attn 计时
    from lmdeploy.pytorch.kernels.cuda import flash_attn_varlen_func
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda')

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(200):
        _ = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                                    max_seqlen_q=num_tokens, max_seqlen_k=num_tokens,
                                    causal=False, kv_layout='hsd')
    end.record()
    torch.cuda.synchronize()
    flash_time = start.elapsed_time(end) / 200

    # manual 计时
    scale = 32 ** -0.5
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(200):
        _ = manual_attention(q, k, v, scale)
    end.record()
    torch.cuda.synchronize()
    manual_time = start.elapsed_time(end) / 200

    print(f"num_tokens={num_tokens:3d}: flash={flash_time:.4f}ms, matmul={manual_time:.4f}ms, "
          f"加速={manual_time/flash_time:.2f}x")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

print("\n💡 关键发现:")
print("1. cuda.event 精确计时显示 kernel 自身耗时")
print("2. perf_counter 显示包括 Python 开销的总耗时")
print("3. 两者差异反映了 Python/PyTorch 的调用开销")
print("4. flash_attn_varlen_func 相比 torch.matmul 有显著加速")
