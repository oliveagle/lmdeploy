#!/usr/bin/env python3
"""
DFlash 推理 Benchmark
运行约 30 秒，测试 DFlash speculative decoding 性能
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'WARNING'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

DURATION = 30  # 运行 30 秒

gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)

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

    # Enable DFlash speculative decoding
    speculative_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
        quant_policy=0,
    )

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tp=1,
        cache_max_entry_count=0.8,  # 80% 的可用显存
        quant_policy=8,
        session_len=16384,  # 支持较长上下文
    )

    print("创建 Pipeline...")
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='WARNING'
    )
    print("✓ 创建成功！\n")

    start_time = time.time()
    total_tokens = 0
    total_requests = 0
    prompt_idx = 0

    print(f"{'序号':<6} {'耗时(s)':<10} {'输出Tokens':<12} {'速度(tokens/s)':<15}")
    print("-" * 50)

    while time.time() - start_time < DURATION:
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1

        t0 = time.time()
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

        print(f"{total_requests:<6} {elapsed:<10.3f} {output_tokens:<12} {tokens_per_sec:<15.2f}")

        if total_requests % 10 == 0:
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / total_requests
            print(f"  -> 当前平均: {avg_time:.3f}s/请求, {total_tokens/elapsed_total:.2f} tokens/s")

    total_time = time.time() - start_time

    # 获取 DFlash 统计信息
    try:
        dflash_stats = pipe.async_engine.engine.get_dflash_stats()
        if dflash_stats and dflash_stats.get('total_draft_tokens', 0) > 0:
            accept_rate = dflash_stats['total_accepted_tokens'] / dflash_stats['total_draft_tokens'] * 100.0
            mal = 1 + dflash_stats['total_accepted_tokens'] / dflash_stats['total_draft_steps'] if dflash_stats.get('total_draft_steps', 0) > 0 else 0
        else:
            accept_rate = 0.0
            mal = 0.0
    except Exception as e:
        print(f"\n无法获取 DFlash 统计: {e}")
        dflash_stats = {}
        accept_rate = 0.0
        mal = 0.0

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