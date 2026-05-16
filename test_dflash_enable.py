#!/usr/bin/env python3
"""
检查 DFlash 是否在推理时保持启用状态
"""

import os
import sys

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'INFO'
os.environ['TM_LOG_LEVEL'] = 'INFO'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tp=1,
    cache_max_entry_count=0.4,
    quant_policy=8,
    session_len=4096,
    speculative_config=SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
        quant_policy=0,
    ),
)

print("创建 Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    log_level='WARNING'
)
print("✓ Pipeline 创建完成\n")

# 获取 engine 并检查 DFlash 状态
engine = pipe.async_engine.engine
print(f"Engine 类型: {type(engine)}")
print(f"Engine 地址: {id(engine)}")

# 检查 model_comm (C++ 绑定)
print(f"model_comm 地址: {id(engine.model_comm)}")

# 获取 DFlash 统计 (初始化)
stats = engine.get_dflash_stats(0)
print(f"初始 DFlash 统计: {stats}")

# 进行一次推理
gen_config = GenerationConfig(max_new_tokens=10, do_sample=False)
print("\n开始推理...")
resp = pipe(
    [{"role": "user", "content": "1+1=?"}],
    gen_config=gen_config,
    sequence_start=True,
    sequence_end=True,
    chat_template_kwargs={'enable_thinking': False}
)
print(f"输出: {resp.text}")

# 获取推理后的 DFlash 统计
stats_after = engine.get_dflash_stats(0)
print(f"\n推理后 DFlash 统计: {stats_after}")

# 检查是否有变化
print(f"\nDraft tokens: {stats_after.get('total_draft_tokens', 0)}")
print(f"Accepted tokens: {stats_after.get('total_accepted_tokens', 0)}")
print(f"Accept rate: {stats_after.get('accept_rate', 0.0) * 100:.2f}%")
