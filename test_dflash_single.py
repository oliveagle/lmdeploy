#!/usr/bin/env python3
"""
DFlash 推理测试 - 单次请求
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'WARNING'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)


def main():
    print("=" * 60)
    print("DFlash 推理测试 - 单次请求")
    print("=" * 60)

    speculative_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
        quant_policy=0,
    )

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tp=1,
        cache_max_entry_count=0.4,
        quant_policy=8,
        session_len=16384,  # 增加 session_len
    )

    print("创建 Pipeline...")
    try:
        pipe = pipeline(
            target_model,
            backend_config=tm_config,
            speculative_config=speculative_config,
            log_level='WARNING'
        )
        print("✓ Pipeline 创建成功！\n")
    except Exception as e:
        print(f"✗ Pipeline 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("执行推理...")
    t0 = time.time()
    try:
        resp = pipe(
            [{"role": "user", "content": "Python 是什么？"}],
            gen_config=gen_config,
            sequence_start=True,
            sequence_end=True,
            chat_template_kwargs={'enable_thinking': False}
        )
        t1 = time.time()

        elapsed = t1 - t0
        print(f"\n✓ 推理完成!")
        print(f"  耗时: {elapsed:.3f}s")
        print(f"  输出长度: {len(resp.token_ids)} tokens")
        print(f"  输出: {resp.text[:200]}...")
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()

    # 获取 DFlash 统计
    try:
        dflash_stats = pipe.async_engine.engine.get_dflash_stats()
        print(f"\nDFlash 统计:")
        print(f"  draft_steps: {dflash_stats.get('total_draft_steps', 0)}")
        print(f"  draft_tokens: {dflash_stats.get('total_draft_tokens', 0)}")
        print(f"  accepted: {dflash_stats.get('total_accepted_tokens', 0)}")
        if dflash_stats.get('total_draft_tokens', 0) > 0:
            ar = dflash_stats['total_accepted_tokens'] / dflash_stats['total_draft_tokens'] * 100
            print(f"  accept_rate: {ar:.1f}%")
    except Exception as e:
        print(f"\n无法获取 DFlash 统计: {e}")

    del pipe
    print("\nDone!")


if __name__ == "__main__":
    main()