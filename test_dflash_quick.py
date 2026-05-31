#!/usr/bin/env python3
"""
DFlash 快速测试 - 只运行一次，查看日志
"""
import os
os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'DEBUG'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

speculative_config = SpeculativeConfig(
    method='dflash',
    model=draft_model,
    num_speculative_tokens=8,
    quant_policy=0,
)

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tp=1,
    cache_max_entry_count=0.2,
    quant_policy=8,
    session_len=16384,
)

print("创建 Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    speculative_config=speculative_config,
    log_level='DEBUG'
)
print("✓ 创建成功！\n")

gen_config = GenerationConfig(max_new_tokens=20, do_sample=False)

print("运行推理...")
resp = pipe(
    [{"role": "user", "content": "What is 2+2?"}],
    gen_config=gen_config,
    sequence_start=True,
    sequence_end=True,
    chat_template_kwargs={'enable_thinking': False}
)

print(f"\n输出: {resp.text}")
print(f"输出 tokens: {len(resp.token_ids) - resp.input_token_len}")

# Get DFlash stats
try:
    tm = pipe.async_engine.engine.model_comm
    stats = tm.get_dflash_stats(0)
    print(f"\nDFlash Stats:")
    print(f"  Draft steps: {stats['total_draft_steps']}")
    print(f"  Draft tokens: {stats['total_draft_tokens']}")
    print(f"  Accepted: {stats['total_accepted_tokens']}")
    print(f"  Rejected: {stats['total_rejected_tokens']}")
    print(f"  Accept rate: {stats['accept_rate']*100:.1f}%")
except Exception as e:
    print(f"DFlash stats error: {e}")

del pipe
