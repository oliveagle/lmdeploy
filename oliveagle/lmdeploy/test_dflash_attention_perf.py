#!/usr/bin/env python3
"""测试 DFlash Attention 性能优化 - PyTorch SDPA fallback"""

import sys
import torch
import time
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

from transformers import AutoConfig
from lmdeploy.pytorch.models.dflash import DFlashAttention

print("=" * 60)
print("测试 DFlashAttention 性能 (V100 SM_70)")
print("=" * 60)

# 检查 GPU
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

# 加载配置
draft_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'
config = AutoConfig.from_pretrained(draft_path)

# 创建 DFlashAttention
attn = DFlashAttention(config, dtype=torch.float16, device='cuda')
print(f"\n✅ DFlashAttention 创建成功")
print(f"   num_heads: {attn.num_heads}")
print(f"   head_dim: {attn.head_dim}")

# 测试参数
num_tokens_list = [8, 16, 32, 64]
warmup = 5
iters = 50

print("\n" + "-" * 60)
print("性能测试")
print("-" * 60)

for num_tokens in num_tokens_list:
    # 创建输入
    hidden_states = torch.randn(num_tokens, config.hidden_size, dtype=torch.float16, device='cuda')
    target_hidden = torch.randn(num_tokens, config.hidden_size, dtype=torch.float16, device='cuda')

    head_dim = config.hidden_size // config.num_attention_heads
    cos = torch.randn(num_tokens, head_dim, dtype=torch.float16, device='cuda')
    sin = torch.randn(num_tokens, head_dim, dtype=torch.float16, device='cuda')
    rotary_pos_emb = (cos, sin)

    # Warmup
    for _ in range(warmup):
        _ = attn(hidden_states, target_hidden, rotary_pos_emb)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iters):
        output = attn(hidden_states, target_hidden, rotary_pos_emb)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_ms = (elapsed / iters) * 1000
    throughput = (num_tokens / (elapsed / iters)) / 1000  # K tokens/s

    print(f"\nnum_tokens={num_tokens:3d}: {avg_ms:7.3f}ms/iter, {throughput:6.1f}K tokens/s")

print("\n" + "=" * 60)
print("✅ 测试完成！")
print("=" * 60)
print("\n💡 说明:")
print("- Triton kernel 可能会在 V100 上失败")
print("- 如果失败，会自动 fallback 到 PyTorch SDPA")
print("- PyTorch SDPA 使用优化实现 (Flash Attention 或 xformers)")
print("- 这比纯 PyTorch 实现快很多")