#!/usr/bin/env python3
"""
DFlash 调试脚本 - 检查详细的日志输出
"""

import os
os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['TM_LOG_LEVEL'] = 'INFO'  # 确保日志级别为 INFO

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash 详细调试 - 检查 [DFlash] 日志")
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
print("开始测试 - 检查关键日志")
print("=" * 60)

gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)
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

print("\n" + "=" * 60)
print("关键日志检查清单：")
print("=" * 60)
print("✓ [DFlash] Enabled DFlash on LanguageModel decoder")
print("✓ [DFlash] Found selected_token_pos (或说明没有)")
print("✓ [DFlash] Collected aux hidden state at layer X")
print("✓ [DFlash] Phase=0: attempting speculative decoding")
print("✓ [DFlash] Draft tokens generated: count=8")
print("✓ [DFlash] VerifyDraft: accepted=N tokens")
print("✓ [DFlash] Success: N accepted draft tokens")
print("=" * 60)