#!/usr/bin/env python3
"""
DFlash Speculative Decoding Performance Benchmark

对比测试：DFlash 开启 vs 关闭时的性能差异

目标:
- DFlash 关闭: ~45 tok/s (baseline)
- DFlash 开启: >= 80 tok/s
- 每 request 显示 DFlash 统计信息

Usage:
    python benchmark_dflash_compare.py [--dflash-off | --dflash-on | --compare]
"""

import argparse
import os
import sys
import time

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

# Model paths
TARGET_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

# Benchmark settings
DURATION = 30  # 运行 30 秒
NUM_SPEC_TOKENS = 8

GEN_CONFIG = GenerationConfig(max_new_tokens=128, do_sample=False)

PROMPTS = [
    "Python 是什么？",
    "人工智能的应用有哪些？",
    "Git 的基本命令有哪些？",
    "什么是深度学习？",
    "Docker 有什么优势？",
]


def create_tm_config(dflash_enabled=True):
    """Create TurboMind config with optional DFlash."""
    config = TurbomindEngineConfig(
        model_format='awq',
        tp=1,
        cache_max_entry_count=0.2,
        quant_policy=8,
        session_len=16384,
    )
    if dflash_enabled:
        config.speculative_config = SpeculativeConfig(
            method='dflash',
            model=DRAFT_MODEL,
            num_speculative_tokens=NUM_SPEC_TOKENS,
            quant_policy=0,
        )
    return config


def run_benchmark(pipe, use_dflash=False, duration=DURATION):
    """Run benchmark and return statistics."""
    start_time = time.time()
    total_tokens = 0
    total_requests = 0
    prompt_idx = 0
    dflash_stats_history = []

    print(f"\n{'='*70}")
    if use_dflash:
        print(f"DFlash 开启 - 运行时长: {duration} 秒")
    else:
        print(f"DFlash 关闭 (Baseline) - 运行时长: {duration} 秒")
    print(f"{'='*70}")

    print(f"\n{'序号':<6} {'耗时(s)':<10} {'输出Tokens':<12} {'速度(t/s)':<10} {'接受率':<10}")
    if use_dflash:
        print(f"{'':6} {'':10} {'':12} {'':10} {'(dflash)':<10}")
    print("-" * 60)

    while time.time() - start_time < duration:
        prompt = PROMPTS[prompt_idx % len(PROMPTS)]
        prompt_idx += 1

        t0 = time.time()
        resp = pipe(
            [{"role": "user", "content": prompt}],
            gen_config=GEN_CONFIG,
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

        # Get DFlash stats if enabled
        accept_rate_str = ""
        if use_dflash:
            try:
                stats = pipe.async_engine.engine.get_dflash_stats(0)
                if stats:
                    draft_tokens = stats.get('total_draft_tokens', 0)
                    accepted = stats.get('total_accepted_tokens', 0)
                    if draft_tokens > 0:
                        rate = accepted / draft_tokens
                        accept_rate_str = f"{rate:.1%}"
                        dflash_stats_history.append(stats)
            except Exception as e:
                accept_rate_str = f"err:{e}"

        print(f"{total_requests:<6} {elapsed:<10.3f} {output_tokens:<12} {tokens_per_sec:<10.2f} {accept_rate_str:<10}")

        # Every 10 requests, show accumulated stats
        if use_dflash and total_requests % 10 == 0:
            _print_dflash_summary(dflash_stats_history)

    total_time = time.time() - start_time
    avg_speed = total_tokens / total_time

    return {
        'total_requests': total_requests,
        'total_tokens': total_tokens,
        'total_time': total_time,
        'avg_speed': avg_speed,
        'dflash_stats': dflash_stats_history,
        'use_dflash': use_dflash,
    }


def _print_dflash_summary(stats_history):
    """Print DFlash stats summary."""
    if not stats_history:
        return

    latest = stats_history[-1]
    print(f"\n  [DFlash 统计汇总]")
    print(f"  {'='*40}")
    print(f"  Draft Steps:   {latest.get('total_draft_steps', 0)}")
    print(f"  Draft Tokens: {latest.get('total_draft_tokens', 0)}")
    print(f"  Accepted:     {latest.get('total_accepted_tokens', 0)}")
    print(f"  Rejected:     {latest.get('total_rejected_tokens', 0)}")
    if latest.get('total_draft_tokens', 0) > 0:
        rate = latest['total_accepted_tokens'] / latest['total_draft_tokens']
        print(f"  Accept Rate:  {rate:.1%}")
    if 'accept_rate' in latest:
        print(f"  Speedup:      {latest['accept_rate']:.1%}")
    print(f"  {'='*40}\n")


def print_result(result):
    """Print benchmark result."""
    print(f"\n{'='*70}")
    print("Benchmark 结果")
    print(f"{'='*70}")
    print(f"DFlash:      {'开启' if result['use_dflash'] else '关闭'}")
    print(f"总请求数:    {result['total_requests']}")
    print(f"总 Tokens:   {result['total_tokens']}")
    print(f"总耗时:      {result['total_time']:.2f}s")
    print(f"平均耗时:    {result['total_time']/result['total_requests']:.3f}s/请求")
    print(f"平均速度:    {result['avg_speed']:.2f} tokens/s")
    print(f"{'='*70}")


def print_comparison(baseline, dflash):
    """Print comparison between baseline and DFlash results."""
    print(f"\n{'='*70}")
    print("性能对比")
    print(f"{'='*70}")
    print(f"{'指标':<20} {'Baseline':<15} {'DFlash':<15} {'差异':<15}")
    print(f"{'-'*60}")
    print(f"{'平均速度 (tok/s)':<20} {baseline['avg_speed']:<15.2f} {dflash['avg_speed']:<15.2f} "
          f"{dflash['avg_speed']-baseline['avg_speed']:+.2f}")
    print(f"{'总请求数':<20} {baseline['total_requests']:<15} {dflash['total_requests']:<15} "
          f"{dflash['total_requests']-baseline['total_requests']:+d}")
    print(f"{'总输出 Tokens':<20} {baseline['total_tokens']:<15} {dflash['total_tokens']:<15} "
          f"{dflash['total_tokens']-baseline['total_tokens']:+d}")

    # DFlash specific stats
    if dflash['dflash_stats']:
        latest = dflash['dflash_stats'][-1]
        draft_tokens = latest.get('total_draft_tokens', 0)
        accepted = latest.get('total_accepted_tokens', 0)
        rejected = latest.get('total_rejected_tokens', 0)

        print(f"\n{'='*70}")
        print("DFlash 详细统计")
        print(f"{'='*70}")
        print(f"Draft Steps:    {latest.get('total_draft_steps', 0)}")
        print(f"Draft Tokens:   {draft_tokens}")
        print(f"Accepted:       {accepted}")
        print(f"Rejected:       {rejected}")
        if draft_tokens > 0:
            accept_rate = accepted / draft_tokens
            speedup = dflash['avg_speed'] / baseline['avg_speed']
            print(f"Accept Rate:    {accept_rate:.1%}")
            print(f"Speedup:        {speedup:.2f}x")

    # Check if targets are met
    print(f"\n{'='*70}")
    print("目标检查")
    print(f"{'='*70}")
    baseline_met = 40 <= baseline['avg_speed'] <= 50
    dflash_met = dflash['avg_speed'] >= 80

    print(f"Baseline (~45 tok/s):  {'✓' if baseline_met else '✗'} {baseline['avg_speed']:.2f} tok/s")
    print(f"DFlash ON (>=80 tok/s): {'✓' if dflash_met else '✗'} {dflash['avg_speed']:.2f} tok/s")

    if baseline_met and dflash_met:
        print("\n✓ 所有性能目标达成!")
    else:
        print("\n⚠ 部分性能目标未达成")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='DFlash Performance Benchmark')
    parser.add_argument('--dflash-off', action='store_true', help='只运行 baseline (DFlash 关闭)')
    parser.add_argument('--dflash-on', action='store_true', help='只运行 DFlash 开启测试')
    parser.add_argument('--compare', action='store_true', default=True, help='对比测试 (默认)')
    parser.add_argument('--duration', type=int, default=DURATION, help=f'运行时长 (秒，默认 {DURATION})')
    args = parser.parse_args()

    # Default to compare mode
    if not (args.dflash_off or args.dflash_on):
        args.compare = True

    print(f"{'='*70}")
    print("DFlash Speculative Decoding 性能对比测试")
    print(f"{'='*70}")
    print(f"目标模型: {TARGET_MODEL}")
    print(f"Draft 模型: {DRAFT_MODEL}")
    print(f"Speculative tokens: {NUM_SPEC_TOKENS}")
    print(f"运行时长: {args.duration} 秒/测试")
    print(f"{'='*70}")

    # Check model paths
    if not os.path.exists(TARGET_MODEL):
        print(f"ERROR: Target model not found at {TARGET_MODEL}")
        sys.exit(1)
    if not os.path.exists(DRAFT_MODEL):
        print(f"ERROR: Draft model not found at {DRAFT_MODEL}")
        sys.exit(1)

    baseline_result = None
    dflash_result = None

    # Run baseline (DFlash OFF)
    if args.compare or args.dflash_off:
        print("\n" + "="*70)
        print("阶段 1: Baseline 测试 (DFlash 关闭)")
        print("="*70)

        tm_config = create_tm_config(dflash_enabled=False)
        print("创建 Pipeline...")
        pipe = None
        try:
            pipe = pipeline(TARGET_MODEL, backend_config=tm_config, log_level='INFO')
            baseline_result = run_benchmark(pipe, use_dflash=False, duration=args.duration)
            print_result(baseline_result)
        except Exception as e:
            print(f"\n[ERROR] Baseline 测试失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if pipe is not None:
                del pipe
        print("\n清理显存...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2)

    # Run DFlash ON
    if args.compare or args.dflash_on:
        print("\n" + "="*70)
        print("阶段 2: DFlash 测试 (DFlash 开启)")
        print("="*70)

        tm_config = create_tm_config(dflash_enabled=True)
        print("创建 Pipeline...")
        pipe = None
        try:
            pipe = pipeline(TARGET_MODEL, backend_config=tm_config, log_level='INFO')
            dflash_result = run_benchmark(pipe, use_dflash=True, duration=args.duration)
            print_result(dflash_result)
        except Exception as e:
            print(f"\n[ERROR] DFlash 测试失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if pipe is not None:
                del pipe

    # Compare results
    if args.compare and baseline_result and dflash_result:
        print_comparison(baseline_result, dflash_result)
    elif baseline_result and not dflash_result:
        print("\n⚠ DFlash 测试未运行，无法进行对比")
    elif dflash_result and not baseline_result:
        print("\n⚠ Baseline 测试未运行，无法进行对比")


if __name__ == "__main__":
    main()
