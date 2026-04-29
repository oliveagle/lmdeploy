#!/usr/bin/env python3
"""Test Qwen3.5-9B with transformers directly."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='cuda',
    trust_remote_code=True,
)

prompt = 'What is the capital of France?'
print(f'\nPrompt: {prompt}')
print('=' * 60)

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'\nResponse:\n{decoded}')
