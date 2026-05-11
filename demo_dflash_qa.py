#!/usr/bin/env python3
"""
DFlash TurboMind 问答 Demo
使用 Qwen3.5-9B-AWQ (target) + Qwen3.5-9B-DFlash (draft) 进行推理加速
"""

import os
import sys
import time

# Set LD_LIBRARY_PATH for the build
os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

# 设置 PyTorch 内存分配器优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

# Model paths
target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash TurboMind 问答 Demo")
print("=" * 60)
print(f"\n目标模型: {target_model}")
print(f"草稿模型: {draft_model}")
print()

# Configure speculative decoding with DFlash
speculative_config = SpeculativeConfig(
    method='dflash',
    model=draft_model,
    num_speculative_tokens=8,
    quant_policy=0,  # FP16
)

# Create TurboMind engine config
tm_config = TurbomindEngineConfig(
    model_format='awq',
    tensor_parallel=1,
    cache_max_entry_count=0.2,  # 降低到 20% 避免 OOM
    quant_policy=8,  # 启用 KV cache 8bit 量化
    session_len=8192,  # 降低最大 session 长度
)

print("正在创建 DFlash Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    speculative_config=speculative_config,
    log_level='WARNING',  # 减少 log 输出
)
print("✓ Pipeline 创建成功！\n")

# Q&A Prompts
qa_prompts = [
    "什么是深度学习？请用一句话解释。",
    "Python 和 JavaScript 的主要区别是什么？",
    "请解释一下什么是 HTTP 状态码 404。",
    "什么是 Git？它的主要用途是什么？",
    "请解释一下什么是容器化技术（如 Docker）。",
]

gen_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=False,  # 使用 greedy 以获得更确定的输出
    temperature=0.7,
    top_p=0.9,
)

# Run inference on each prompt
print("=" * 60)
print("开始问答测试")
print("=" * 60)

total_time = 0
for i, prompt in enumerate(qa_prompts, 1):
    print(f"\n【问题 {i}】")
    print("-" * 60)
    print(f"问题: {prompt}")
    print("\n回答:")

    start_time = time.time()
    # 使用消息格式并禁用思考模式
    # Qwen3.5 使用 Linear Attention，每个请求必须是独立的（stateless）
    messages = [{"role": "user", "content": prompt}]
    response = pipe(
        messages,
        gen_config=gen_config,
        sequence_start=True,  # 标记为独立请求的开始
        sequence_end=True,    # 标记为独立请求的结束
        chat_template_kwargs={'enable_thinking': False}
    )
    elapsed = time.time() - start_time
    total_time += elapsed

    print(response.text)
    print(f"\n⏱️  耗时: {elapsed:.2f}秒")
    print("-" * 60)

print("\n" + "=" * 60)
print(f"✓ 所有问答完成！总耗时: {total_time:.2f}秒")
print(f"✓ 平均每题耗时: {total_time/len(qa_prompts):.2f}秒")
print("=" * 60)