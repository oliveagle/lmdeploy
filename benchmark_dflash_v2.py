#!/usr/bin/env python3
"""
DFlash 推理 Benchmark - 简化版
使用更小的 session_len 避免 warmup 失败
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'INFO'
os.environ['TM_LOG_LEVEL'] = 'INFO'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

DURATION = 30  # 运行 30 秒

# 使用较小的 max_new_tokens 避免 warmup 问题
gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)

prompts = [
    "Python 是什么？",
    "人工智能的应用有哪些？",
    "Git 的基本命令有哪些？",
    "什么是深度学习？",
    "Docker 有什么优势？",
]


def main():
    print("=" * 60)
    print("DFlash 推理 Benchmark")
    print("=" * 60)
    print(f"目标: {target_model}")
    print(f"草稿: {draft_model}")
    print(f"运行时长: {DURATION} 秒\n")

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tp=1,
        cache_max_entry_count=0.4,
        quant_policy=8,
        session_len=4096,  # 增加 session_len 以避免 truncation
        speculative_config=SpeculativeConfig(
            method='dflash',
            model=draft_model,
            num_speculative_tokens=8,
            quant_policy=0,
        ),
    )

    print("创建 Pipeline...")
    try:
        pipe = pipeline(
            target_model,
            backend_config=tm_config,
            log_level='WARNING'
        )
        print("✓ 创建成功！\n")
    except Exception as e:
        print(f"✗ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    start_time = time.time()
    total_tokens = 0
    total_requests = 0
    prompt_idx = 0
    last_draft_tokens = 0
    last_accepted_tokens = 0
    running_accept_rate = 0.0

    print(f"{'序号':<6} {'耗时(s)':<10} {'输出Tokens':<12} {'速度(tokens/s)':<15} {'Draft接受率':<12}")
    print("-" * 70)

    while time.time() - start_time < DURATION:
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1

        t0 = time.time()
        try:
            resp = pipe(
                [{"role": "user", "content": prompt}],
                gen_config=gen_config,
                sequence_start=True,
                sequence_end=True,
                chat_template_kwargs={'enable_thinking': False}
            )
            t1 = time.time()

            elapsed = t1 - t0
            output_tokens = len(resp.token_ids) - resp.input_token_len
            tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0
            total_tokens += output_tokens
            total_requests += 1

            # 获取 DFlash 接受率
            accept_rate_str = "N/A"
            try:
                stats = pipe.async_engine.engine.get_dflash_stats()
                if stats and stats.get('total_draft_tokens', 0) > 0:
                    current_draft = stats['total_draft_tokens']
                    current_accepted = stats['total_accepted_tokens']
                    # 计算本次请求的增量接受率
                    draft_delta = current_draft - last_draft_tokens
                    accept_delta = current_accepted - last_accepted_tokens
                    if draft_delta > 0:
                        accept_rate_str = f"{accept_delta / draft_delta * 100:.1f}%"
                    else:
                        accept_rate_str = "0.0%"
                    last_draft_tokens = current_draft
                    last_accepted_tokens = current_accepted
                    # 更新运行平均
                    running_accept_rate = current_accepted / current_draft * 100
            except Exception:
                pass

            print(f"{total_requests:<6} {elapsed:<10.3f} {output_tokens:<12} {tokens_per_sec:<15.2f} {accept_rate_str:<12}")

            if total_requests % 10 == 0:
                elapsed_total = time.time() - start_time
                avg_time = elapsed_total / total_requests
                print(f"  -> 当前平均: {avg_time:.3f}s/请求, {total_tokens/elapsed_total:.2f} tokens/s")
        except Exception as e:
            print(f"请求失败: {e}")
            break

    total_time = time.time() - start_time

    # 获取 DFlash 统计信息
    dflash_stats = {}
    accept_rate = 0.0
    mal = 0.0
    try:
        dflash_stats = pipe.async_engine.engine.get_dflash_stats()
        if dflash_stats and dflash_stats.get('total_draft_tokens', 0) > 0:
            accept_rate = dflash_stats['total_accepted_tokens'] / dflash_stats['total_draft_tokens'] * 100.0
            mal = 1 + dflash_stats['total_accepted_tokens'] / dflash_stats['total_draft_steps'] if dflash_stats.get('total_draft_steps', 0) > 0 else 0
    except Exception as e:
        print(f"\n无法获取 DFlash 统计: {e}")

    print("\n" + "=" * 60)
    print("Benchmark 结果")
    print("=" * 60)
    print(f"总请求数:   {total_requests}")
    print(f"总 Tokens:  {total_tokens}")
    print(f"总耗时:     {total_time:.2f}s")
    print(f"平均耗时:   {total_time/total_requests:.3f}s/请求")
    print(f"平均速度:   {total_tokens/total_time:.2f} tokens/s")
    print("\n" + "-" * 60)
    print("DFlash Speculative Decoding 统计")
    print("-" * 60)
    print(f"总 draft 步数:       {dflash_stats.get('total_draft_steps', 0)}")
    print(f"总 draft tokens:     {dflash_stats.get('total_draft_tokens', 0)}")
    print(f"总 accepted tokens: {dflash_stats.get('total_accepted_tokens', 0)}")
    print(f"Draft 接受率:        {accept_rate:.2f}%")
    print(f"平均接受长度:        {mal:.2f}")
    print("=" * 60)

    del pipe


if __name__ == "__main__":
    main()