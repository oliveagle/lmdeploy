#!/usr/bin/env python3
"""
真实测量 Python TurboMind Prefill 性能 - Qwen3.6-35B-A3B-AWQ
使用正确的 prefill 优化配置
"""

import os
import sys
import time
import asyncio
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
WARMUP_RUNS = 2
MEASURE_RUNS = 5

REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count: int) -> str:
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats

TEST_CONTEXTS = {
    "1K": 1000,
    "2K": 2000,
    "4K": 4000,
    "8K": 8000,
}


async def run_benchmark():
    print("=" * 80)
    print("Python TurboMind Prefill 性能测试 - Qwen3.6-35B-A3B-AWQ")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    print(f"Warmup: {WARMUP_RUNS}, Measure: {MEASURE_RUNS}")

    # 创建引擎 - 关键优化配置
    print("\n[1/3] 创建引擎（启用 prefill 优化）...", end=" ", flush=True)
    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=16384,
            max_batch_size=1,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            model_format='awq',
            cache_max_entry_count=0.4,
            # 关键 Prefill 优化参数
            max_prefill_token_num=16384,   # 允许大 prefill
            num_tokens_per_iter=16384,      # 一次处理所有 token
            max_prefill_iters=1,            # 只跑一轮
        ),
    )
    tok = Tokenizer(MODEL)
    inst = tm.create_instance()
    print("OK")

    # 打印引擎配置
    print(f"\n引擎配置:")
    print(f"  max_prefill_token_num: {tm.engine_config.max_prefill_token_num}")
    print(f"  num_tokens_per_iter:   {tm.engine_config.num_tokens_per_iter}")
    print(f"  max_prefill_iters:     {tm.engine_config.max_prefill_iters}")

    # Warmup
    print(f"\n[2/3] Warmup ({WARMUP_RUNS} runs)...", end=" ", flush=True)
    warmup_ids = tok.encode(gen_prompt(2048))
    for i in range(WARMUP_RUNS):
        async for _ in inst.async_stream_infer(
            session_id=i, input_ids=warmup_ids,
            gen_config=GenerationConfig(max_new_tokens=1),
            sequence_start=True, sequence_end=True,
        ):
            pass
    print("OK")

    # 测量
    print(f"[3/3] 测量预填充性能 ({MEASURE_RUNS} runs)...")
    results = {}
    gen_cfg = GenerationConfig(max_new_tokens=1, temperature=0.7)

    print(f"\n{'Context':>8} | {'Tokens':>8} | {'Avg TTFT':>10} | {'Min TTFT':>10} | "
          f"{'Avg TPS':>12} | {'Max TPS':>12}")
    print("-" * 80)

    for label, target_tokens in TEST_CONTEXTS.items():
        prompt = gen_prompt(target_tokens)
        input_ids = tok.encode(prompt)
        actual_tokens = len(input_ids)
        run_times = []

        print(f"{label:>8} ({actual_tokens:>6} tok)... ", end="", flush=True)

        for r in range(MEASURE_RUNS):
            session_id = r + 100000 + hash(label) % 100000
            start = time.perf_counter()

            try:
                async for out in inst.async_stream_infer(
                    session_id=session_id, input_ids=input_ids,
                    gen_config=gen_cfg,
                    sequence_start=True, sequence_end=True,
                ):
                    if out.status.value in (1, 2):
                        elapsed = (time.perf_counter() - start) * 1000
                        run_times.append(elapsed)
                        print(f"{elapsed:.0f}ms ", end="", flush=True)
                        break
            except Exception as e:
                print(f"\nERROR: {e}", end="", flush=True)
                break

        if run_times:
            avg_ms = sum(run_times) / len(run_times)
            min_ms = min(run_times)
            max_ms = max(run_times)
            avg_tps = (actual_tokens / avg_ms * 1000) if avg_ms > 0 else 0
            max_tps = (actual_tokens / min_ms * 1000) if min_ms > 0 else 0

            results[label] = {
                "ctx_tokens": actual_tokens,
                "avg_ms": avg_ms, "min_ms": min_ms, "max_ms": max_ms,
                "avg_tps": avg_tps, "max_tps": max_tps,
            }
            print(f"| Avg {avg_tps:>8.0f} tok/s, Max {max_tps:>8.0f} tok/s (TTFT: {avg_ms:.1f}ms)")
        else:
            print("| ERROR - no results")

    # 总结
    print("\n" + "=" * 80)
    print("总结 - Qwen3.6-35B-A3B-AWQ Prefill 性能")
    print("=" * 80)
    for label in ["1K", "2K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"  {label:>6}: {r['avg_tps']:>10.0f} tok/s (TTFT: {r['avg_ms']:>8.1f}ms)")

    # 保存结果
    import json
    output = {
        "model": MODEL,
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "measure_runs": MEASURE_RUNS,
            "max_prefill_token_num": 16384,
            "num_tokens_per_iter": 16384,
            "max_prefill_iters": 1,
        },
        "results": results,
        "timestamp": time.time(),
    }

    out_path = Path(__file__).parent / "prefill_benchmark_real.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    tm.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
