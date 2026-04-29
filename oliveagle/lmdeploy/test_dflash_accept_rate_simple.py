#!/usr/bin/env python3
"""DFlash Accept Rate Test with Metrics"""
import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import time
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from lmdeploy.messages import SpeculativeConfig

if __name__ == '__main__':

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
        log_level='WARNING',  # 减少日志输出
    )

    print("✅ Pipeline 创建成功\n")

    # 测试 prompts
    prompts = [
        "What is the capital of France?",
        "Write a hello world in Python.",
        "Explain what is a neural network.",
    ]

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:50]}...")

        start_time = time.time()
        output = pipe(prompt, GenerationConfig(max_new_tokens=128, temperature=0.0))
        elapsed = time.time() - start_time
        num_tokens = len(output.token_ids)

        print(f"  Tokens: {num_tokens}, Time: {elapsed:.2f}s, Speed: {num_tokens/elapsed:.1f} t/s")
        print(f"  Output preview: {output.text[:100]}...")
        print()

        results.append({'tokens': num_tokens, 'time': elapsed, 'speed': num_tokens/elapsed})

    total_tokens = sum(r['tokens'] for r in results)
    total_time = sum(r['time'] for r in results)
    avg_speed = total_tokens / total_time if total_time > 0 else 0

    print("=" * 50)
    print(f"总 tokens: {total_tokens}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均速度: {avg_speed:.1f} tokens/s")
    print("✅ 测试完成!")
