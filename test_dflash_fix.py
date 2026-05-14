#!/usr/bin/env python3
"""
DFlash Speculative Decoding 修复验证

测试修复后的 DFlash 实现，验证：
1. DFlash 在 decode 阶段运行
2. Draft tokens 正确存储和验证
3. Accept rate 指标
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'INFO'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

DURATION = 30
NUM_SPEC_TOKENS = 8

gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)

prompts = [
    "Python 是什么？",
    "人工智能的应用有哪些？",
    "Git 的基本命令有哪些？",
    "什么是深度学习？",
    "Docker 有什么优势？",
]

def main():
    print("=" * 70)
    print("DFlash Speculative Decoding 修复验证")
    print("=" * 70)
    print(f"目标模型: {target_model}")
    print(f"草稿模型: {draft_model}")
    print(f"Speculative tokens: {NUM_SPEC_TOKENS}")
    print(f"运行时长: {DURATION} 秒\n")

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
    )

    print("创建 Pipeline...")
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='INFO'
    )
    print("✓ 创建成功！\n")

    print("预期行为:")
    print("  - DFlash 在 decode 阶段运行（global_token_num == 1）")
    print("  - Draft tokens 跨迭代存储和验证")
    print("  - 性能应该高于无 DFlash 的基线 (~45 tokens/s)\n")

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

            print(f"{total_requests:<6} {elapsed:<10.3f} {output_tokens:<12} {tokens_per_sec:<15.2f}")

            # 每 10 次请求打印一次统计
            if total_requests % 10 == 0:
                elapsed_total = time.time() - start_time
                print(f"  -> 当前平均: {elapsed_total/total_requests:.3f}s/请求, {total_tokens/elapsed_total:.2f} tokens/s")

                # Try to get DFlash stats from the engine
                try:
                    tm = pipe.async_engine.engine.model_comm
                    dflash_stats = tm.get_dflash_stats(0)
                    print(f"  -> DFlash Stats:")
                    print(f"     - Draft steps: {dflash_stats['total_draft_steps']}")
                    print(f"     - Draft tokens: {dflash_stats['total_draft_tokens']}")
                    print(f"     - Accepted: {dflash_stats['total_accepted_tokens']}")
                    print(f"     - Rejected: {dflash_stats['total_rejected_tokens']}")
                    print(f"     - Accept rate: {dflash_stats['accept_rate']*100:.1f}%")
                except Exception as e:
                    print(f"  -> DFlash stats not available: {e}")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Benchmark 结果")
    print("=" * 70)
    print(f"总请求数:   {total_requests}")
    print(f"总 Tokens:  {total_tokens}")
    print(f"总耗时:     {total_time:.2f}s")
    print(f"平均耗时:   {total_time/total_requests:.3f}s/请求")
    print(f"平均速度:   {total_tokens/total_time:.2f} tokens/s")

    # Final DFlash stats
    try:
        tm = pipe.async_engine.engine.model_comm
        dflash_stats = tm.get_dflash_stats(0)
        print("\n" + "=" * 70)
        print("DFlash 统计")
        print("=" * 70)
        print(f"Draft steps:     {dflash_stats['total_draft_steps']}")
        print(f"Draft tokens:    {dflash_stats['total_draft_tokens']}")
        print(f"Accepted tokens: {dflash_stats['total_accepted_tokens']}")
        print(f"Rejected tokens: {dflash_stats['total_rejected_tokens']}")
        print(f"Accept rate:     {dflash_stats['accept_rate']*100:.1f}%")
    except Exception as e:
        print(f"\n[INFO] DFlash stats not available: {e}")

    print("\n" + "=" * 70)
    print("修复验证")
    print("=" * 70)
    baseline_speed = 45.0  # 无 DFlash 的基线速度
    current_speed = total_tokens / total_time
    improvement = (current_speed - baseline_speed) / baseline_speed * 100

    if current_speed > baseline_speed:
        print(f"✓ 成功！当前速度 {current_speed:.2f} tokens/s > 基线 {baseline_speed} tokens/s")
        print(f"  提升: {improvement:.1f}%")
    else:
        print(f"⚠ 当前速度 {current_speed:.2f} tokens/s < 基线 {baseline_speed} tokens/s")
        print(f"  差距: {improvement:.1f}%")
        print("  注意: Accept rate 越高，加速效果越明显")

    print("\n日志中应该看到:")
    print("  - '[DFlash] === DECODE MODE: Verifying draft tokens ==='")
    print("  - '[DFlash] Found stored draft tokens'")
    print("  - '[DFlash] VerifyDraft: accepted=X tokens'")
    print("=" * 70)

    del pipe

if __name__ == "__main__":
    main()
