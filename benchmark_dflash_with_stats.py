#!/usr/bin/env python3
"""
DFlash 推理 Benchmark (带接受率统计)
"""

import os
import time
import sys

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'INFO'  # INFO to see DFlash logs without too much noise

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

DURATION = 30  # 运行 30 秒
NUM_SPEC_TOKENS = 8  # speculative tokens per step

gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)

prompts = [
    "Python 是什么？",
    "人工智能的应用有哪些？",
    "Git 的基本命令有哪些？",
    "什么是深度学习？",
    "Docker 有什么优势？",
]

# DFlash 统计变量
dflash_stats = {
    'total_draft_steps': 0,      # 总 draft 步数
    'total_draft_tokens': 0,     # 总 draft tokens 生成
    'total_accepted_tokens': 0,  # 总接受的 tokens
    'total_iterations': 0,       # 总迭代次数
}

def main():
    print("=" * 60)
    print("DFlash 推理 Benchmark (带接受率统计)")
    print("=" * 60)
    print(f"目标: {target_model}")
    print(f"草稿: {draft_model}")
    print(f"运行时长: {DURATION} 秒")
    print(f"Speculative tokens: {NUM_SPEC_TOKENS}\n")

    # Enable DFlash speculative decoding
    speculative_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=NUM_SPEC_TOKENS,
        quant_policy=0,
    )

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tp=1,
        cache_max_entry_count=0.2,
        quant_policy=8,
        session_len=16384,
        enable_metrics=True,  # 启用 metrics 来获取 DFlash 统计
    )

    print("创建 Pipeline...")
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='INFO'
    )
    print("✓ 创建成功！\n")

    start_time = time.time()
    total_tokens = 0
    total_requests = 0
    prompt_idx = 0

    print(f"{'序号':<6} {'耗时(s)':<10} {'输出Tokens':<12} {'速度(tokens/s)':<15}")
    print("-" * 50)

    try:
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

            # 更新 DFlash 统计 (估算)
            # 每次 decode 迭代可能生成 NUM_SPEC_TOKENS 个 draft tokens
            total_iterations += 1
            dflash_stats['total_iterations'] += 1
            # 假设每步都尝试 speculation，draft tokens = iterations * spec_tokens
            dflash_stats['total_draft_steps'] += 1
            dflash_stats['total_draft_tokens'] += NUM_SPEC_TOKENS

            print(f"{total_requests:<6} {elapsed:<10.3f} {output_tokens:<12} {tokens_per_sec:<15.2f}")

            # 每 10 次请求打印一次统计
            if total_requests % 10 == 0:
                elapsed_total = time.time() - start_time
                # 估算接受率: 输出 tokens / 总 iterations
                # 每个 iteration 最多接受 NUM_SPEC_TOKENS 个 tokens
                max_possible_accepted = total_iterations * NUM_SPEC_TOKENS
                estimated_accept_rate = (total_tokens / max_possible_accepted * 100) if max_possible_accepted > 0 else 0

                print(f"  -> 迭代次数: {total_iterations}, Draft tokens: {dflash_stats['total_draft_tokens']}")
                print(f"  -> 估算接受率: {estimated_accept_rate:.1f}% (tokens/iterations)")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Benchmark 结果")
    print("=" * 60)
    print(f"总请求数:   {total_requests}")
    print(f"总 Tokens:  {total_tokens}")
    print(f"总耗时:     {total_time:.2f}s")
    print(f"平均耗时:   {total_time/total_requests:.3f}s/请求")
    print(f"平均速度:   {total_tokens/total_time:.2f} tokens/s")

    print("\n" + "=" * 60)
    print("DFlash 统计 (估算)")
    print("=" * 60)
    print(f"迭代次数:           {dflash_stats['total_iterations']}")
    print(f"Draft tokens 总数:    {dflash_stats['total_draft_tokens']}")
    print(f"最大可能接受 tokens: {dflash_stats['total_iterations'] * NUM_SPEC_TOKENS}")

    # 计算接受率
    max_possible = dflash_stats['total_iterations'] * NUM_SPEC_TOKENS
    accept_rate = (total_tokens / max_possible * 100) if max_possible > 0 else 0
    print(f"估算接受率:         {accept_rate:.1f}%")
    print(f"平均每次迭代接受:   {total_tokens / dflash_stats['total_iterations']:.1f} tokens (max {NUM_SPEC_TOKENS})")
    print("=" * 60)

    # 获取实际的 metrics（如果可用）
    try:
        metrics = pipe.async_engine.engine.get_schedule_metrics()
        print(f"\n引擎 Metrics: {metrics}")
    except:
        pass

    del pipe

if __name__ == "__main__":
    main()
