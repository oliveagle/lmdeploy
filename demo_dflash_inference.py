#!/usr/bin/env python3
"""
DFlash 性能对比测试 - 第二阶段：DFlash 推理
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("阶段 2: DFlash 推理")
print("=" * 60)
print(f"\n目标模型: {target_model}")
print(f"草稿模型: {draft_model}")

speculative_config = SpeculativeConfig(
    method='dflash',
    model=draft_model,
    num_speculative_tokens=8,
    quant_policy=0,
)

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tensor_parallel=1,
    cache_max_entry_count=0.2,
    quant_policy=8,
    session_len=16384,  # 增加到 16k，避免溢出
)

print("\n创建 DFlash Pipeline...")
pipe = pipeline(target_model, backend_config=tm_config,
                speculative_config=speculative_config, log_level='WARNING')
print("✓ Pipeline 创建成功！\n")

gen_config = GenerationConfig(max_new_tokens=256, do_sample=False)
prompts = [
    "人工智能是什么？",
    "Python 的特点是什么？",
    "Git 是什么？",
]

times = []
for i, prompt in enumerate(prompts, 1):
    print(f"--- {i}/{len(prompts)}: {prompt}")
    t0 = time.time()
    resp = pipe([{"role": "user", "content": prompt}], gen_config=gen_config,
                sequence_start=True, sequence_end=True,
                chat_template_kwargs={'enable_thinking': False})
    t1 = time.time()
    times.append(t1 - t0)
    print(f"回答: {resp.text[:80]}...")
    print(f"耗时: {t1-t0:.2f}秒\n")

print(f"平均耗时: {sum(times)/len(times):.2f}秒")