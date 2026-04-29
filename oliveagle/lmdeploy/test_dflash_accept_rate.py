#!/usr/bin/env python3
"""
简单的 DFlash speculative decoding 接受率测试
"""

import sys
import os
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from lmdeploy.messages import SpeculativeConfig
import time

print("=" * 70)
print("DFlash Speculative Decoding 接受率测试")
print("=" * 70)

# 模型路径
target_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
draft_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'

# 测试提示
prompts = [
    "Write a Python quicksort function.",
    "What is the capital of France?",
    "Explain the difference between TCP and UDP.",
]

# Speculative 配置
spec_config = SpeculativeConfig(
    method='dflash',
    model=draft_model_path,
    num_speculative_tokens=8,
)

# Backend 配置
backend_config = PytorchEngineConfig(
    dtype='float16',
    cache_max_entry_count=0.5,
    eager_mode=True,  # 使用 eager mode 避免 CUDA graph 问题
)

# 创建 pipeline
print(f"\n创建 pipeline...")
print(f"Target: {target_model_path}")
print(f"Draft: {draft_model_path}")
print(f"Spec tokens: 8")
print(f"Eager mode: True")

pipe = pipeline(
    target_model_path,
    speculative_config=spec_config,
    backend_config=backend_config,
    log_level='INFO',
)

print(f"✅ Pipeline 创建成功")

# 运行测试
results = []
for i, prompt in enumerate(prompts, 1):
    print(f"\n{'='*70}")
    print(f"测试 {i}/{len(prompts)}: {prompt[:50]}...")
    print(f"{'='*70}")

    start_time = time.time()

    output = pipe(
        prompt,
        GenerationConfig(
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        ),
    )

    elapsed = time.time() - start_time
    num_tokens = len(output.token_ids)

    print(f"✅ 完成")
    print(f"  生成 tokens: {num_tokens}")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  速度: {num_tokens/elapsed:.1f} tokens/s")
    print(f"  输出预览: {output.text[:100]}...")

    results.append({
        'prompt': prompt,
        'tokens': num_tokens,
        'time': elapsed,
        'speed': num_tokens/elapsed,
    })

# 打印结果摘要
print(f"\n" + "=" * 70)
print(f"结果摘要")
print("=" * 70)

total_tokens = sum(r['tokens'] for r in results)
total_time = sum(r['time'] for r in results)
avg_speed = total_tokens / total_time if total_time > 0 else 0

print(f"总 tokens: {total_tokens}")
print(f"总耗时: {total_time:.2f}s")
print(f"平均速度: {avg_speed:.1f} tokens/s")

print(f"\n✅ 测试完成!")
