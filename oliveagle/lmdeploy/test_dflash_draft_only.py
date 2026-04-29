#!/usr/bin/env python3
"""Quick DFlash debug"""
import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import torch
from transformers import AutoConfig
from lmdeploy.pytorch.models.dflash import DFlashForCausalLM
from lmdeploy.pytorch.model_inputs import StepContextManager

# Setup
draft_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'
config = AutoConfig.from_pretrained(draft_path)
print(f'Config loaded, architecture: {config.architectures}')
print(f'dflash_config: {getattr(config, "dflash_config", None)}')

ctx_mgr = StepContextManager()
model = DFlashForCausalLM(config, ctx_mgr, device='cuda', dtype=torch.float16)
print(f'Draft model created, num params: {sum(p.numel() for p in model.parameters()):,}')

# Mock input
num_spec = 8
vocab = config.vocab_size
input_ids = torch.tensor([[config.pad_token_id] * num_spec], dtype=torch.int64, device='cuda')
position_ids = torch.arange(num_spec, device='cuda').unsqueeze(0)

# Target hidden
dflash_cfg = getattr(config, 'dflash_config', None)
target_layer_ids = None
if dflash_cfg is not None:
    if isinstance(dflash_cfg, dict):
        target_layer_ids = dflash_cfg.get('target_layer_ids', [1,8,15,22,29])
    else:
        target_layer_ids = getattr(dflash_cfg, 'target_layer_ids', [1,8,15,22,29])
if not target_layer_ids:
    target_layer_ids = [1,8,15,22,29]
num_target_layers = len(target_layer_ids)
target_hidden_size = num_target_layers * config.hidden_size

target_hidden = torch.randn((1, num_spec, target_hidden_size), dtype=torch.float16, device='cuda')
print(f'target_hidden shape: {target_hidden.shape}')

print('Forward start')
out = model(input_ids, position_ids, [], target_hidden_states=target_hidden)
print(f'Output hidden shape: {out.shape}')

print('Get logits')
logits = model.get_logits(out)
print(f'Logits shape: {logits.shape}')

draft_tokens = logits.argmax(dim=-1)
print(f'Draft tokens: {draft_tokens}')
