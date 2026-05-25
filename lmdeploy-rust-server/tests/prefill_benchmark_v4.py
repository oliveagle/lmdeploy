#!/usr/bin/env python3
"""
TurboMind Prefill 性能分析 v4 - 完整测量
"""

import os, sys, time
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
WARMUP = 3
MEASURE = 5

REPEAT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(n):
    return REPEAT * max(1, n // 10)

def run():
    print("=" * 80)
    print("TurboMind Prefill 性能分析 v4")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Warmup: {WARMUP}, Measure: {MEASURE}")
    
    pipe = pipeline(
        MODEL,
        backend_config=TurbomindEngineConfig(
            session_len=16384, max_batch_size=1, cache_block_seq_len=64,
            tp=1, enable_prefix_caching=False, model_format='awq',
            cache_max_entry_count=0.4,
        ),
        log_level='ERROR',
    )
    
    print(f"\n{'Context':>8} | {'TTFT (ms)':>12} | {'TPS':>15} | {'All runs':>40}")
    print("-" * 85)
    
    results = {}
    for label, ctx in [("1K", 1024), ("2K", 2048), ("4K", 4096), ("8K", 8192)]:
        prompt = gen_prompt(ctx)
        cfg = GenerationConfig(max_new_tokens=1, temperature=0.7, do_sample=False)
        
        # Warmup - 不同 session_id 避免 cache
        for i in range(WARMUP):
            for _ in pipe.stream_infer([prompt], gen_config=cfg, do_preprocess=False, stream_response=True):
                pass
        
        # Measure
        times = []
        for i in range(MEASURE):
            t0 = time.perf_counter()
            for resp in pipe.stream_infer([prompt], gen_config=cfg, do_preprocess=False, stream_response=True):
                if resp.generate_token_len >= 1:
                    times.append((time.perf_counter() - t0) * 1000)
                    break
        
        if times:
            avg = sum(times) / len(times)
            tps = ctx / avg * 1000
            results[label] = {"ttft": avg, "tps": tps, "times": times}
            runs_str = ", ".join([f"{t:.0f}ms" for t in times])
            print(f"{label:>8} | {avg:>10.1f}   | {tps:>10.0f} tok/s | {runs_str}")
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    for label in ["1K", "2K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"  {label:>6}: {r['tps']:>8.0f} tok/s (TTFT: {r['ttft']:.1f}ms)")
    
    # 对比历史
    print(f"\n历史数据 (2026-05-18):")
    print(f"  1K:   14,827 tok/s (TTFT: 69ms)")
    print(f"  4K:   33,727 tok/s (TTFT: 122ms)")
    print(f"  8K:   42,875 tok/s (TTFT: 191ms)")
    
    print(f"\n当前 vs 历史:")
    for label, hist in [("1K", 14827), ("4K", 33727), ("8K", 42875)]:
        if label in results:
            curr = results[label]['tps']
            ratio = hist / curr
            print(f"  {label:>6}: {curr:>8.0f} vs {hist:>8.0} = {ratio:>4.1f}x 慢")

if __name__ == "__main__":
    run()
