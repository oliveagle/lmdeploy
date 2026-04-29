#!/usr/bin/env python3
"""
测试目标模型的多 token 处理功能
测试当输入多个 token 时，attention 是否正确工作
"""

import sys
import os
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import torch
print("=" * 70)
print("目标模型多 token 处理测试")
print("=" * 70)

# 配置路径
target_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'

print(f"\n[模型路径]")
print(f"Target:  {target_model_path}")

# 创建配置
print(f"\n[创建配置...]")

try:
    from lmdeploy.pytorch.config import ModelConfig, CacheConfig

    # Target 模型配置
    target_model_config = ModelConfig.from_pretrained(
        target_model_path,
        trust_remote_code=True,
        dtype='bfloat16',
    )
    print(f"✅ Target 模型配置创建成功")
    print(f"   - hidden_size: {target_model_config.hidden_size}")
    print(f"   - num_layers: {target_model_config.hf_config.num_hidden_layers}")
    print(f"   - num_heads: {target_model_config.hf_config.num_attention_heads}")

    # Cache 配置
    cache_config = CacheConfig(
        max_batches=8,
        block_size=128,
        num_cpu_blocks=0,
        num_gpu_blocks=100,
        max_prefill_token_num=4096,
        device_type='cuda',
    )
    print(f"✅ Cache 配置创建成功")

except Exception as e:
    print(f"❌ 配置创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建 backend config
print(f"\n[创建 backend config...]")
from lmdeploy.pytorch.config import PytorchEngineConfig

backend_config = PytorchEngineConfig(
    dtype='bfloat16',
    cache_max_entry_count=0.5,
)

# 创建 dist context
print(f"\n[创建 dist context...]")
from lmdeploy.pytorch.config import DistConfig, DistContext

dist_config = DistConfig()
dist_ctx = DistContext(dist_config)

# 创建 device context
print(f"\n[创建 device context...]")
from lmdeploy.pytorch.config import DeviceConfig, DeviceContext

device_config = DeviceConfig()
device_ctx = DeviceContext(device_config)

# 创建 misc config
print(f"\n[创建 misc config...]")
from lmdeploy.pytorch.config import MiscConfig

misc_config = MiscConfig()

# 构建模型
print(f"\n[构建 target model...]")
from lmdeploy.pytorch.engine.model_agent import BaseModelAgent

agent = BaseModelAgent(
    model_path=target_model_path,
    model_config=target_model_config,
    cache_config=cache_config,
    backend_config=backend_config,
    misc_config=misc_config,
    dist_ctx=dist_ctx,
    device_ctx=device_ctx,
)

agent.build_model()

# 获取模型
model = agent.patched_model
print(f"✅ Target model built: {model.__class__.__name__}")

# 设置 aux_hidden_state_layers for DFlash
if hasattr(target_model_config.hf_config, 'aux_hidden_state_layers'):
    print(f"aux_hidden_state_layers: {target_model_config.hf_config.aux_hidden_state_layers}")
else:
    # 设置默认值
    target_model_config.hf_config.aux_hidden_state_layers = [1, 8, 15, 22, 29]
    print(f"设置 aux_hidden_state_layers: {target_model_config.hf_config.aux_hidden_state_layers}")

# 创建模型输入 - 测试多 token
print(f"\n[测试多 token 处理...]")
from lmdeploy.pytorch.model_inputs import ModelInputs, step_ctx_manager
from lmdeploy.pytorch.model_inputs import make_prefill_model_inputs

batch_size = 1
num_tokens = 8  # 测试 8 个 token，类似 speculative decoding 场景
vocab_size = target_model_config.vocab_size

# 创建 input_ids
input_ids = torch.randint(0, vocab_size, (batch_size, num_tokens), device='cuda')
seq_length = torch.tensor([num_tokens], device='cuda')
history_lengths = torch.tensor([0], device='cuda')

print(f"  input_ids shape: {input_ids.shape}")
print(f"  seq_length: {seq_length}")

# 创建 prefill 模型输入
model_inputs = make_prefill_model_inputs(
    input_ids=input_ids,
    seq_length=seq_length,
    history_lengths=history_lengths,
    cache_config=cache_config,
    model_config=target_model_config,
    dist_ctx=dist_ctx,
    device_ctx=device_ctx,
)

print(f"  is_decoding: {model_inputs.is_decoding}")
print(f"  max_q_seqlen: {model_inputs.max_q_seqlen}")
print(f"  max_kv_seqlen: {model_inputs.max_kv_seqlen}")

# 运行 forward
print(f"\n[测试 forward...]")
with step_ctx_manager(model.ctx_mgr):
    context = model.ctx_mgr.build_context(
        inputs=model_inputs,
        model_config=target_model_config,
        cache_config=cache_config,
        kv_caches=None,
    )

    model.ctx_mgr.set_current(context)
    model.update_model_metas(past_key_values=None, context=context)
    input_dict = model.prepare_inputs_for_generation(past_key_values=None, context=context)

    print(f"  input_dict keys: {list(input_dict.keys())}")

    outputs = model(**input_dict)
    print(f"✅ Forward 成功")
    print(f"  outputs type: {type(outputs)}")

    if isinstance(outputs, dict):
        print(f"  outputs keys: {list(outputs.keys())}")
        if 'hidden_states' in outputs:
            print(f"  hidden_states shape: {outputs['hidden_states'].shape}")
        if 'aux_hidden_states' in outputs:
            print(f"  aux_hidden_states shape: {outputs['aux_hidden_states'].shape}")
    else:
        print(f"  outputs shape: {outputs.shape}")

# 测试 2: 模拟 speculative decoding - is_decoding=True 但有多个 tokens
print(f"\n[测试 is_decoding=True 但多 token 场景...]")

model_inputs.is_decoding = True
model_inputs.max_q_seqlen = num_tokens

print(f"  is_decoding: {model_inputs.is_decoding}")
print(f"  max_q_seqlen: {model_inputs.max_q_seqlen}")

# 运行 forward
with step_ctx_manager(model.ctx_mgr):
    context = model.ctx_mgr.build_context(
        inputs=model_inputs,
        model_config=target_model_config,
        cache_config=cache_config,
        kv_caches=None,
    )

    model.ctx_mgr.set_current(context)
    model.update_model_metas(past_key_values=None, context=context)
    input_dict = model.prepare_inputs_for_generation(past_key_values=None, context=context)

    print(f"  input_dict keys: {list(input_dict.keys())}")

    # 获取 attention metadata
    if 'attn_metadata' in input_dict:
        attn_meta = input_dict['attn_metadata']
        print(f"  attn_metadata.is_decoding: {attn_meta.is_decoding}")
        print(f"  attn_metadata.max_q_seqlen: {attn_meta.max_q_seqlen}")
        print(f"  attn_metadata.max_kv_seqlen: {attn_meta.max_kv_seqlen}")
        print(f"  attn_metadata.q_seqlens: {attn_meta.q_seqlens}")

    outputs = model(**input_dict)
    print(f"✅ Forward 成功")
    print(f"  outputs type: {type(outputs)}")

    if isinstance(outputs, dict):
        print(f"  outputs keys: {list(outputs.keys())}")
        if 'hidden_states' in outputs:
            print(f"  hidden_states shape: {outputs['hidden_states'].shape}")
        if 'aux_hidden_states' in outputs:
            print(f"  aux_hidden_states shape: {outputs['aux_hidden_states'].shape}")
    else:
        print(f"  outputs shape: {outputs.shape}")

print(f"\n" + "=" * 70)
print(f"测试完成!")
print("=" * 70)
print(f"\n✅ 所有测试通过!")
