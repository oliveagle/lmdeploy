#!/usr/bin/env python3
"""测试方案1: DFlash 使用 flash_attn_varlen_func (避免缓存命中)

使用随机输入避免缓存命中，测试真实性能
"""

import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import torch
import time

print("=" * 60)
print("测试方案1: DFlash 真实性能测试 (随机输入)")
print("=" * 60)

print(f"\n[环境]")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 导入
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

# 测试不同 num_tokens 的性能
print("\n[性能测试] 使用随机输入 (每次迭代数据不同)")
print("-" * 60)

results = []

for num_tokens in [1, 8, 16, 32, 64]:
    hidden_size = 128

    # Warmup (固定数据)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    target_hidden = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    cos = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')
    sin = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')

    with torch.no_grad():
        for _ in range(10):
            _ = attn(hidden_states, target_hidden, (cos, sin))
    torch.cuda.synchronize()

    # Benchmark (随机数据，避免缓存)
    iters = 100
    start = time.time()

    for _ in range(iters):
        # 每次迭代使用新的随机数据
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
        target_hidden = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
        cos = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')
        sin = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')

        with torch.no_grad():
            output = attn(hidden_states, target_hidden, (cos, sin))

    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_ms = elapsed / iters * 1000
    throughput = num_tokens * iters / elapsed

    results.append((num_tokens, avg_ms, throughput))
    print(f"num_tokens={num_tokens:2d}: {avg_ms:6.3f}ms/iter, {throughput:8.0f} tokens/sec")

# 对比测试：固定数据 vs 随机数据
print("\n[对比测试] 固定数据 vs 随机数据 (num_tokens=64)")
print("-" * 60)

num_tokens = 64
iters = 100

# 固定数据 (可能缓存命中)
hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
target_hidden = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
cos = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')
sin = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')

with torch.no_grad():
    for _ in range(10):
        _ = attn(hidden_states, target_hidden, (cos, sin))
torch.cuda.synchronize()

start = time.time()
for _ in range(iters):
    with torch.no_grad():
        _ = attn(hidden_states, target_hidden, (cos, sin))
torch.cuda.synchronize()
fixed_elapsed = time.time() - start

# 随机数据 (无缓存)
start = time.time()
for _ in range(iters):
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    target_hidden = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    cos = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')
    sin = torch.randn(num_tokens, config.head_dim, dtype=torch.float16, device='cuda')
    with torch.no_grad():
        _ = attn(hidden_states, target_hidden, (cos, sin))
torch.cuda.synchronize()
random_elapsed = time.time() - start

print(f"固定数据: {fixed_elapsed/iters*1000:.3f}ms/iter, {num_tokens*iters/fixed_elapsed:.0f} tokens/sec")
print(f"随机数据: {random_elapsed/iters*1000:.3f}ms/iter, {num_tokens*iters/random_elapsed:.0f} tokens/sec")
print(f"差异: {random_elapsed/fixed_elapsed:.2f}x")

# 内存带宽分析
print("\n[内存带宽分析]")
print("-" * 60)
num_tokens = 64
hidden_size = 128

# 估算内存访问量
# Q/K/V: 64 * 128 * 2 bytes (fp16) * 3 (Q,K,V) = 49KB
# 输出: 64 * 128 * 2 = 16KB
# 总计: ~65KB per iteration
bytes_per_iter = num_tokens * hidden_size * 2 * 4  # Q,K,V,O
bandwidth_gbps = bytes_per_iter / (random_elapsed / iters) / 1e9

print(f"每次迭代内存访问: ~{bytes_per_iter/1024:.1f} KB")
print(f"实测带宽: {bandwidth_gbps:.1f} GB/s")
print(f"V100 理论带宽: ~900 GB/s (HBM2)")
print(f"带宽利用率: {bandwidth_gbps/900*100:.1f}%")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

print("\n💡 结论:")
print("1. 随机数据测试显示真实性能")
print("2. 小 batch size (1-16 tokens) 时 kernel launch 开销较大")
print("3. 大 batch size (32-64 tokens) 时 GPU 利用率更高")
print("4. 带宽利用率较低是因为计算量小，受 kernel launch 延迟限制")
