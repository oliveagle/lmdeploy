#!/usr/bin/env python3
"""DFlash Accept Rate Test - 简单版本"""
import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

if __name__ == '__main__':
    from lmdeploy.messages import SpeculativeConfig
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

    target_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
    draft_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'

    spec_config = SpeculativeConfig(
        method='dflash',
        model=draft_model_path,
        num_speculative_tokens=8,
    )

    backend_config = PytorchEngineConfig(
        dtype='float16',
        cache_max_entry_count=0.5,
        eager_mode=True,
    )

    print("创建 pipeline...")
    pipe = pipeline(
        target_model_path,
        speculative_config=spec_config,
        backend_config=backend_config,
        log_level='INFO',
    )

    prompts = [
        "What is the capital of France?",
        "Write a hello world in Python.",
        "Explain what is a neural network.",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt[:40]}...")
        output = pipe(prompt, GenerationConfig(max_new_tokens=64, temperature=0.0))
        print(f"  输出: {output.text[:100]}...")
        print(f"  Tokens: {len(output.token_ids)}")

    print("\n✅ 测试完成!")
