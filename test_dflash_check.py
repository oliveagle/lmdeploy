#!/usr/bin/env python3
"""
检查 DFlash 是否真正启用
"""

import os
import sys

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'DEBUG'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("检查 DFlash 配置")
print("=" * 60)

# 检查配置结构
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

print(f"\nTurbomindEngineConfig.speculative_config:")
print(f"  method: {tm_config.speculative_config.method}")
print(f"  model: {tm_config.speculative_config.model}")
print(f"  num_speculative_tokens: {tm_config.speculative_config.num_speculative_tokens}")

print("\n创建 Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    log_level='INFO'
)

print("\n检查 Pipeline 结构:")
print(f"  pipe 类型: {type(pipe)}")
print(f"  有 async_engine: {hasattr(pipe, 'async_engine')}")

if hasattr(pipe, 'async_engine'):
    print(f"  async_engine 类型: {type(pipe.async_engine)}")
    print(f"  有 engine: {hasattr(pipe.async_engine, 'engine')}")

    if hasattr(pipe.async_engine, 'engine'):
        print(f"  engine 类型: {type(pipe.async_engine.engine)}")
        print(f"  有 get_dflash_stats: {hasattr(pipe.async_engine.engine, 'get_dflash_stats')}")

        # 尝试调用 get_dflash_stats
        try:
            stats = pipe.async_engine.engine.get_dflash_stats(0)
            print(f"\n  get_dflash_stats() 返回: {stats}")
        except Exception as e:
            print(f"\n  get_dflash_stats() 失败: {e}")

# 执行一次推理
print("\n" + "=" * 60)
print("执行推理测试...")
print("=" * 60)

gen_config = GenerationConfig(max_new_tokens=32, do_sample=False)

resp = pipe(
    [{"role": "user", "content": "你好"}],
    gen_config=gen_config,
    sequence_start=True,
    sequence_end=True,
    chat_template_kwargs={'enable_thinking': False}
)

print(f"\n输入 tokens: {resp.input_token_len}")
print(f"输出 tokens: {len(resp.token_ids) - resp.input_token_len}")
print(f"输出文本: {resp.text}")

# 再次检查统计
if hasattr(pipe, 'async_engine') and hasattr(pipe.async_engine, 'engine'):
    try:
        stats = pipe.async_engine.engine.get_dflash_stats(0)
        print(f"\n推理后 DFlash 统计: {stats}")

        if stats:
            draft_tokens = stats.get('total_draft_tokens', 0)
            accepted = stats.get('total_accepted_tokens', 0)
            print(f"\nDraft tokens: {draft_tokens}")
            print(f"Accepted: {accepted}")
            if draft_tokens > 0:
                print(f"接受率: {accepted / draft_tokens * 100:.2f}%")
            else:
                print("⚠️  没有 draft tokens - DFlash 可能未启用!")
    except Exception as e:
        print(f"获取统计失败: {e}")

del pipe
