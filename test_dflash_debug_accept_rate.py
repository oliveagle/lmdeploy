#!/usr/bin/env python3
"""
调试 DFlash 接受率问题
"""

import os
import sys
import time

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'DEBUG'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 70)
print("DFlash 调试 - 检查接受率")
print("=" * 70)

# 创建配置
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

print(f"\n配置检查:")
print(f"  speculative_config.method: {tm_config.speculative_config.method}")
print(f"  speculative_config.model: {tm_config.speculative_config.model}")
print(f"  speculative_config.num_speculative_tokens: {tm_config.speculative_config.num_speculative_tokens}")

print("\n创建 Pipeline...")
pipe = pipeline(
    target_model,
    backend_config=tm_config,
    log_level='INFO'
)

print(f"\nPipeline 结构检查:")
print(f"  pipe 类型: {type(pipe)}")
print(f"  有 async_engine: {hasattr(pipe, 'async_engine')}")

if not hasattr(pipe, 'async_engine'):
    print("\n❌ ERROR: pipe 没有 async_engine 属性!")
    sys.exit(1)

print(f"  async_engine 类型: {type(pipe.async_engine)}")
print(f"  有 engine: {hasattr(pipe.async_engine, 'engine')}")

if not hasattr(pipe.async_engine, 'engine'):
    print("\n❌ ERROR: async_engine 没有 engine 属性!")
    sys.exit(1)

print(f"  engine 类型: {type(pipe.async_engine.engine)}")
print(f"  有 get_dflash_stats: {hasattr(pipe.async_engine.engine, 'get_dflash_stats')}")

if not hasattr(pipe.async_engine.engine, 'get_dflash_stats'):
    print("\n❌ ERROR: engine 没有 get_dflash_stats 方法!")
    sys.exit(1)

print("\n✓ Pipeline 结构正确")

# 测试推理
gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)

prompts = [
    "Python 是什么？",
    "什么是深度学习？",
    "Git 有什么用？",
]

print("\n" + "=" * 70)
print("开始推理测试")
print("=" * 70)

for i, prompt in enumerate(prompts, 1):
    print(f"\n--- 请求 {i}: {prompt} ---")

    t0 = time.time()
    resp = pipe(
        [{"role": "user", "content": prompt}],
        gen_config=gen_config,
        sequence_start=True,
        sequence_end=True,
        chat_template_kwargs={'enable_thinking': False}
    )
    t1 = time.time()

    output_tokens = len(resp.token_ids) - resp.input_token_len
    elapsed = t1 - t0

    print(f"输入 tokens: {resp.input_token_len}")
    print(f"输出 tokens: {output_tokens}")
    print(f"耗时: {elapsed:.3f}s")
    print(f"速度: {output_tokens / elapsed:.2f} tokens/s")

    # 获取 DFlash 统计
    try:
        stats = pipe.async_engine.engine.get_dflash_stats(0)
        print(f"\nDFlash 统计: {stats}")

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

                # 验证数据一致性
                expected_total = accepted + rejected
                if expected_total != draft_tokens:
                    print(f"  ⚠️  警告: accepted + rejected ({expected_total}) != draft_tokens ({draft_tokens})")
            else:
                print("  ❌ ERROR: draft_tokens = 0，DFlash 可能未启用!")
        else:
            print("  ❌ ERROR: get_dflash_stats() 返回空数据!")
    except Exception as e:
        print(f"  ❌ ERROR: 获取统计失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("调试完成")
print("=" * 70)

del pipe
