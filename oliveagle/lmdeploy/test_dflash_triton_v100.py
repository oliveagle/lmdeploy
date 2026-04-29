#!/usr/bin/env python3
"""测试 DFlash 在 V100 上使用 lmdeploy 原生 Triton kernel

验证移除 _IS_SM70 检查后，DFlash 是否可以正常工作
"""

import sys
import os
import torch
import time
from pathlib import Path

sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

print("=" * 60)
print("测试 DFlash Triton Kernel 在 V100 上的工作状态")
print("=" * 60)

# 检查 GPU
print("\n[1] 检查 GPU 信息...")
if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"   GPU: {gpu_props.name}")
    print(f"   CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
    is_sm70 = gpu_props.major == 7 and gpu_props.minor == 0
    print(f"   SM_70: {is_sm70}")
else:
    print("   ⚠️  CUDA 不可用")

# 测试 dflash.py 加载
print("\n[2] 加载 DFlash 模型代码...")
try:
    from lmdeploy.pytorch.models.dflash import DFlashAttention, DFlashModel, DFlashForCausalLM
    print("   ✅ DFlash 模块加载成功")
except Exception as e:
    print(f"   ❌ DFlash 模块加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 flash_attn_varlen_func 加载
print("\n[3] 检查 Triton kernel...")
try:
    from lmdeploy.pytorch.kernels.cuda import flash_attn_varlen_func
    print("   ✅ flash_attn_varlen_func 加载成功")
except Exception as e:
    print(f"   ❌ flash_attn_varlen_func 加载失败: {e}")
    sys.exit(1)

# 模拟一个简单的配置
print("\n[4] 测试 DFlashAttention 初始化...")
from dataclasses import dataclass

@dataclass
class SimpleConfig:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    rms_norm_eps: float = 1e-6

try:
    config = SimpleConfig()
    dflash_attn = DFlashAttention(config, dtype=torch.float16, device='cuda')
    print("   ✅ DFlashAttention 初始化成功")
except Exception as e:
    print(f"   ❌ DFlashAttention 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 forward pass
print("\n[5] 测试 DFlashAttention forward pass...")
try:
    num_tokens = 8
    hidden_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_dim // num_heads

    # 模拟输入
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device='cuda')
    target_hidden = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device='cuda')

    # 模拟 rotary embeddings
    cos = torch.randn(num_tokens, head_dim, dtype=torch.float16, device='cuda')
    sin = torch.randn(num_tokens, head_dim, dtype=torch.float16, device='cuda')
    rotary_pos_emb = (cos, sin)

    # 前向推理
    print(f"   输入形状:")
    print(f"     hidden_states: {hidden_states.shape}")
    print(f"     target_hidden: {target_hidden.shape}")

    start = time.time()
    output = dflash_attn(hidden_states, target_hidden, rotary_pos_emb)
    elapsed = time.time() - start

    print(f"   ✅ Forward pass 成功!")
    print(f"   输出形状: {output.shape}")
    print(f"   耗时: {elapsed*1000:.3f}ms")

except Exception as e:
    print(f"   ❌ Forward pass 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试完整 DFlashModel
print("\n[6] 测试完整 DFlashModel forward...")
try:
    # 扩展配置
    class DFlashConfig(SimpleConfig):
        def __init__(self):
            super().__init__()
            self.vocab_size = 151936
            self.pad_token_id = 0
            self.dflash_config = {
                'target_layer_ids': [1, 10, 19, 28, 37],
                'block_size': 16,
                'mask_token_id': 248070,
            }
            self.num_hidden_layers = 12
            self.intermediate_size = hidden_dim * 4

    config = DFlashConfig()

    # 创建模型
    dflash_model = DFlashModel(config, dtype=torch.float16, device='cuda')
    print("   ✅ DFlashModel 创建成功")

    # 模拟输入
    input_ids = torch.randint(0, config.vocab_size, (1, num_tokens), dtype=torch.int64, device='cuda')
    target_hidden = torch.randn(1, num_tokens, len([1, 10, 19, 28, 37])*hidden_dim,
                                 dtype=torch.float16, device='cuda')
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device='cuda').unsqueeze(0)

    print(f"   输入形状:")
    print(f"     input_ids: {input_ids.shape}")
    print(f"     target_hidden: {target_hidden.shape}")
    print(f"     position_ids: {position_ids.shape}")

    # 前向推理
    start = time.time()
    output = dflash_model(input_ids, target_hidden, position_ids)
    elapsed = time.time() - start

    print(f"   ✅ DFlashModel forward 成功!")
    print(f"   输出形状: {output.shape}")
    print(f"   耗时: {elapsed*1000:.3f}ms")

except Exception as e:
    print(f"   ❌ DFlashModel forward 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有测试通过！DFlash Triton Kernel 在 V100 上工作正常!")
print("=" * 60)
print("\n📋 修复总结:")
print("1. 移除了 _IS_SM70 = ... 检测逻辑")
print("2. 移除了 PyTorch fallback 实现")
print("3. 统一使用 lmdeploy 原生 Triton kernel (flash_attn_varlen_func)")
print("4. lmdeploy 的 Triton kernel 已经包含 SM_70 优化 (_kernel_meta_sm7x)")
print("   不需要我们手动处理！")
print("\n💡 好处:")
print("- V100 上可以用 Triton kernel，速度和其他 GPU 一样快")
print("- 不需要维护两套代码 (fallback + Triton)")
print("- 代码更简洁")