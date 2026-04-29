#!/usr/bin/env python3
"""
单进程测试 - 验证 DFlash 修复
"""
import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

from lmdeploy.messages import SpeculativeConfig
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
import time

print("=" * 70)
print("单进程 DFlash 测试")
print("=" * 70)

target_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
draft_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'

spec_config = SpeculativeConfig(
    method='dflash',
    model=draft_model_path,
    num_speculative_tokens=8,
)

backend_config = PytorchEngineConfig(
    dtype='bfloat16',
    cache_max_entry_count=0.5,
)

print(f"创建 pipeline...")
pipe = pipeline(
    target_model_path,
    speculative_config=spec_config,
    backend_config=backend_config,
    log_level='INFO',
)

print(f"测试推理...")
prompt = "What is the capital of France?"

start_time = time.time()
output = pipe(
    prompt,
    GenerationConfig(
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    ),
)
elapsed = time.time() - start_time

num_tokens = len(output.token_ids)
print(f"✅ 完成!")
print(f"生成 tokens: {num_tokens}")
print(f"耗时: {elapsed:.2f}s")
print(f"速度: {num_tokens/elapsed:.1f} tokens/s")
print(f"输出: {output.text[:200]}")
