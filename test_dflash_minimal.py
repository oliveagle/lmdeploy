#!/usr/bin/env python3
"""
最小化测试 - 定位段错误
"""

import os
import sys

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'INFO'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("创建配置...")
tm_config = TurbomindEngineConfig(
    model_format='awq',
    tp=1,
    cache_max_entry_count=0.4,
    quant_policy=8,
    session_len=2048,
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
    log_level='INFO'
)

print("Pipeline 创建成功!")

# 检查统计 API
print("\n检查 get_dflash_stats...")
try:
    stats = pipe.async_engine.engine.get_dflash_stats(0)
    print(f"初始统计: {stats}")
except Exception as e:
    print(f"获取统计失败: {e}")

# 测试非常短的输出
print("\n测试推理 (max_new_tokens=16)...")
gen_config = GenerationConfig(max_new_tokens=16, do_sample=False)

try:
    resp = pipe(
        [{"role": "user", "content": "你好"}],
        gen_config=gen_config,
        sequence_start=True,
        sequence_end=True,
        chat_template_kwargs={'enable_thinking': False}
    )
    print(f"推理成功! 输出: {resp.text[:100]}")

    # 再次检查统计
    stats = pipe.async_engine.engine.get_dflash_stats(0)
    print(f"推理后统计: {stats}")
except Exception as e:
    print(f"推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")
del pipe
