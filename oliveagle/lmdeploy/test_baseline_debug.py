#!/usr/bin/env python3
"""Debug lmdeploy baseline output."""
import sys
import os
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

os.environ['LMDEPLOY_SKIP_WARMUP'] = '1'

from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

target_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
prompts = ["What is the capital of France? The capital is"]

backend_config = PytorchEngineConfig(
    dtype='float16',
    cache_max_entry_count=0.5,
    eager_mode=True,
)

pipe = pipeline(
    target_model_path,
    backend_config=backend_config,
    log_level='WARNING',
)

for prompt in prompts:
    print(f"Prompt: {prompt}")
    output = pipe(
        prompt,
        GenerationConfig(
            max_new_tokens=10,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        ),
    )
    print(f"Token IDs: {output.token_ids}")
    print(f"Text: {repr(output.text)}")
