#!/usr/bin/env python3
"""
测试 DFlash 在多个 decode step 中的工作
生成较长的输出来看 speculative decoding 是否生效
"""

import os

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash 长文本生成测试 - 搜索 [DFlash] 相关日志")
print("=" * 60)

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
    session_len=16384,
)

print("\n创建 Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    speculative_config=speculative_config,
    log_level='INFO',
)

print("\n" + "=" * 60)
print("开始长文本生成测试")
print("=" * 60)

# 增加 max_new_tokens 到更大值，看是否能触发 DFlash
gen_config = GenerationConfig(max_new_tokens=512, do_sample=False)

prompt = "请详细介绍人工智能的发展历史，包括重要的里程碑事件。"

messages = [{"role": "user", "content": prompt}]
response = pipe(
    messages,
    gen_config=gen_config,
    sequence_start=True,
    sequence_end=True,
    chat_template_kwargs={'enable_thinking': False}
)

print(f"\n回答 (前300字符): {response.text[:300]}...")
print(f"\n输入 tokens: {response.input_token_len}")
print(f"输出 tokens: {len(response.token_ids) - response.input_token_len}")
print(f"总 tokens: {len(response.token_ids)}")
print("\n检查上方完整日志中是否有以下内容:")
print("  1. [DFlash] Collected aux hidden state")
print("  2. [DFlash] DFlash enabled: enable=1")
print("  3. [DFlash] Output X accepted draft tokens")