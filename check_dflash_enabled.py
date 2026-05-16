#!/usr/bin/env python3
"""
检查 DFlash 是否真的被启用
"""

import os
os.environ['LD_LIBRARY_PATH'] = f'{os.path.dirname(os.path.abspath(__file__))}/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("检查 DFlash 状态")
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

print("\n创建 DFlash Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    speculative_config=speculative_config,
    log_level='DEBUG',  # 使用 DEBUG 级别查看详细信息
)

print("\n" + "=" * 60)
print("测试推理")
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
print(f"输出 tokens: {len(response.token_ids) - response.input_token_len}")

# Get DFlash stats
try:
    stats = pipe.engine.model.comm.get_dflash_stats(0)
    print(f"\nDFlash Stats:")
    print(f"  Draft Steps: {stats.get('total_draft_steps', 0)}")
    print(f"  Draft Tokens: {stats.get('total_draft_tokens', 0)}")
    print(f"  Accepted: {stats.get('total_accepted_tokens', 0)}")
    print(f"  Rejected: {stats.get('total_rejected_tokens', 0)}")
    if stats.get('total_draft_tokens', 0) > 0:
        accept_rate = stats['total_accepted_tokens'] / stats['total_draft_tokens'] * 100
        print(f"  Accept Rate: {accept_rate:.1f}%")
except Exception as e:
    print(f"\nFailed to get DFlash stats: {e}")

print("\n检查日志中的 DFlash 相关信息：")
print("1. [DFlash] 开头的日志")
print("2. dflash_accepted_tokens 相关信息")
print("3. Draft model 加载信息")