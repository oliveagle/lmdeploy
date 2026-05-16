#!/usr/bin/env python3
"""
安全调试 DFlash 接受率 - 不崩溃版本
"""

import os
import sys

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'WARNING'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash 调试 - 安全版")
print("=" * 60)

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tp=1,
    cache_max_entry_count=0.4,
    quant_policy=8,
    session_len=4096,  # 增加 session_len 以避免 truncation
    speculative_config=SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=4,  # 减少到 4 个
        quant_policy=0,
    ),
)

print("创建 Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    log_level='WARNING'
)

print("✓ Pipeline 创建成功!")

# 运行一个推理
gen_config = GenerationConfig(max_new_tokens=32, do_sample=False)

print("\n运行推理...")
try:
    resp = pipe(
        [{"role": "user", "content": "你好"}],
        gen_config=gen_config,
        sequence_start=True,
        sequence_end=True,
        chat_template_kwargs={'enable_thinking': False}
    )
    print(f"✓ 推理成功!")
    print(f"  输出: {resp.text}")
except Exception as e:
    print(f"✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n检查 DFlash 统计...")
try:
    stats = pipe.async_engine.engine.get_dflash_stats(0)
    print(f"  统计: {stats}")

    if stats:
        draft_steps = stats.get('total_draft_steps', 0)
        draft_tokens = stats.get('total_draft_tokens', 0)
        accepted = stats.get('total_accepted_tokens', 0)
        rejected = stats.get('total_rejected_tokens', 0)

        print(f"  Draft 步数: {draft_steps}")
        print(f"  Draft tokens: {draft_tokens}")
        print(f"  Accepted: {accepted}")
        print(f"  Rejected: {rejected}")

        if draft_tokens > 0:
            accept_rate = accepted / draft_tokens * 100
            print(f"  接受率: {accept_rate:.2f}%")
        else:
            print("  ⚠️  没有 draft tokens - 需要检查 DFlash 是否真正执行")
    else:
        print("  ⚠️  统计为空")
except Exception as e:
    print(f"  ✗ 获取统计失败: {e}")

del pipe
print("\n完成!")