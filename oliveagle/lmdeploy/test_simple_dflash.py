#!/usr/bin/env python3
"""Simple DFlash accept rate test"""
import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from lmdeploy.messages import SpeculativeConfig

if __name__ == '__main__':

    target_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
    draft_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'

    spec_config = SpeculativeConfig(method='dflash', model=draft_path, num_speculative_tokens=8)

    backend_config = PytorchEngineConfig(
        dtype='float16',
        cache_max_entry_count=0.5,
        eager_mode=True
    )

    print('Creating pipeline...')
    pipe = pipeline(target_path, speculative_config=spec_config, backend_config=backend_config, log_level='WARNING')

    print('Pipeline created')

    prompts = [
        "What is the capital of France?"
    ]

    for i, prompt in enumerate(prompts):
        print(f'\n=== Prompt {i}: {prompt} ===')
        output = pipe(prompt, GenerationConfig(max_new_tokens=32, temperature=0.0))
        print(f'Output: {output.text}')
        print(f'Generated: {len(output.token_ids)} tokens')
