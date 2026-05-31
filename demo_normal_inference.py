#!/usr/bin/env python3
"""
DFlash 性能对比测试 - 第一阶段：普通推理
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"

print("=" * 60)
print("阶段 1: 普通推理 (无 DFlash)")
print("=" * 60)
print(f"\n目标模型: {target_model}")

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tensor_parallel=1,
    cache_max_entry_count=0.2,
    quant_policy=8,
    session_len=8192,
)

print("\n创建普通 Pipeline...")
pipe = pipeline(target_model, backend_config=tm_config, log_level='WARNING')
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