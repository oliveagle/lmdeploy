#!/usr/bin/env python3
"""
DFlash TurboMind 自然语言推理 Demo
使用 Qwen3.5-9B-AWQ (target) + Qwen3.5-9B-DFlash (draft) 进行推理加速
"""

import os
import sys
import time

# Set LD_LIBRARY_PATH for the build
os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

# Model paths
target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash TurboMind 自然语言推理 Demo")
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
    cache_max_entry_count=0.5,  # 使用更小的 KV cache，0.5 表示空闲显存的 50%
    quant_policy=8,  # 启用 KV cache 8bit 量化，减少内存占用
)

print("正在创建 DFlash Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    speculative_config=speculative_config,
    log_level='WARNING',  # 减少 log 输出
)
print("✓ Pipeline 创建成功！\n")

# Natural Language Reasoning Prompts
nlr_prompts = [
    {
        "category": "逻辑推理",
        "prompt": "如果所有猫都怕水，小黑是猫，小黑怕水吗？"
    },
    {
        "category": "数学推理",
        "prompt": "农场有鸡兔共50只，腿140条。各几只？"
    },
    {
        "category": "因果推理",
        "prompt": "小明迟到因为错过公交。说明公交晚点了吗？"
    },
    {
        "category": "常识推理",
        "prompt": "水倒进有洞的杯子会发生什么？"
    },
    {
        "category": "类比推理",
        "prompt": "医生对病人，老师对什么？"
    },
]

gen_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=False,  # 使用 greedy 以获得更确定的推理结果
    temperature=0.7,
    top_p=0.9,
)

# Run inference on each prompt
print("=" * 60)
print("开始自然语言推理测试")
print("=" * 60)

total_time = 0
for i, item in enumerate(nlr_prompts, 1):
    print(f"\n【测试 {i}】{item['category']}")
    print("-" * 60)
    print(f"问题: {item['prompt']}")
    print("\n推理中...")

    start_time = time.time()
    # 使用消息格式并禁用思考模式
    # Qwen3.5 使用 Linear Attention，每个请求必须是独立的（stateless）
    messages = [{"role": "user", "content": item['prompt']}]
    response = pipe(
        messages,
        gen_config=gen_config,
        sequence_start=True,  # 标记为独立请求的开始
        sequence_end=True,    # 标记为独立请求的结束
        chat_template_kwargs={'enable_thinking': False}
    )
    elapsed = time.time() - start_time
    total_time += elapsed

    print(f"\n回答:")
    print(response.text)
    print(f"\n⏱️  耗时: {elapsed:.2f}秒")
    print("-" * 60)

print("\n" + "=" * 60)
print(f"✓ 所有推理任务完成！总耗时: {total_time:.2f}秒")
print(f"✓ 平均每题耗时: {total_time/len(nlr_prompts):.2f}秒")
print("=" * 60)
