#!/usr/bin/env python3
"""
详细的 DFlash 调试脚本
检查 DFlash 是否真的在工作
"""

import os
os.environ['TM_LOG_LEVEL'] = 'DEBUG'  # 开启详细日志

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash 详细调试")
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
    log_level='INFO',  # 使用 INFO 级别查看 DFlash 日志
)

print("\n" + "=" * 60)
print("推理测试 - 搜索 [DFlash] 关键词")
print("=" * 60)

gen_config = GenerationConfig(max_new_tokens=256, do_sample=False)
prompt = "Python 是什么？"

messages = [{"role": "user", "content": prompt}]
response = pipe(
    messages,
    gen_config=gen_config,
    sequence_start=True,
    sequence_end=True,
    chat_template_kwargs={'enable_thinking': False}
)

print(f"\n回答: {response.text}")
print(f"\n输入 tokens: {response.input_token_len}")
print(f"输出 tokens: {len(response.token_ids) - response.input_token_len}")
print(f"总 tokens: {len(response.token_ids)}")
print("\n检查上方日志中是否有 [DFlash] 开头的行...")