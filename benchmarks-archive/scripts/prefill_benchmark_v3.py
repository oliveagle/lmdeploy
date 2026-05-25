#!/usr/bin/env python3
"""
正确测量 TurboMind C++ Prefill 吞吐量
使用官方 Profiler 方法，直接测量真实的 prefill performance
"""

import os, sys, time, json
from pathlib import Path
import numpy as np

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
WARMUP_RUNS = 2
MEASURE_RUNS = 3

REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count: int) -> str:
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats

TEST_CONTEXTS = {
    "4K": 4000,
    "8K": 8000,
    "16K": 16000,
    # "32K": 32000,  # OOM on 32GB GPU
}


async def run_benchmark():
    print("=" * 80)
    print("TurboMind C++ Prefill 吞吐量测试 - Qwen3.6-35B-A3B-AWQ")
    print("=" * 80)
    print(f"\nModel: {MODEL}")

    # 创建引擎 - 使用大 max_prefill_token_num
    print("\n[1/3] 创建引擎...", end=" ", flush=True)
    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=32768,
            max_batch_size=1,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            max_prefill_token_num=32768,  # 允许大预填充
            num_tokens_per_iter=32768,    # 一次处理所有 token
            max_prefill_iters=1,          # 只跑一轮 prefill
            cache_max_entry_count=0.4,    # 降低 cache 占用给 32K 留空间
            model_format='awq',
        ),
    )
    tok = Tokenizer(MODEL)
    inst = tm.create_instance()
    print("OK")

    # 检查引擎配置
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
          f"{'Avg TPS':>12} | {'Max TPS':>12} | {'Iters':>5}")
    print("-" * 80)

    for label, target_tokens in TEST_CONTEXTS.items():
        prompt = gen_prompt(target_tokens)
        input_ids = tok.encode(prompt)
        actual_tokens = len(input_ids)
        run_times = []
        all_token_counts = []

        print(f"{label:>8} ({actual_tokens:>6} tok)... ", end="", flush=True)

        for r in range(MEASURE_RUNS):
            session_id = r + 2000 + hash(label) % 1000
            start = time.perf_counter()
            token_counts = []

            try:
                async for out in inst.async_stream_infer(
                    session_id=session_id, input_ids=input_ids,
                    gen_config=gen_cfg,
                    sequence_start=True, sequence_end=True,
                ):
                    elapsed = (time.perf_counter() - start) * 1000
                    token_counts.append(len(out.token_ids))

                    if out.status.value in (1, 2):
                        run_times.append(elapsed)
                        all_token_counts.append(token_counts)
                        print(f"{elapsed:.0f}ms({len(token_counts)}iters) ", end="", flush=True)
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
            avg_iters = np.mean([len(tc) for tc in all_token_counts])

            results[label] = {
                "ctx_tokens": actual_tokens,
                "avg_ms": avg_ms, "min_ms": min_ms, "max_ms": max_ms,
                "avg_tps": avg_tps, "max_tps": max_tps,
                "avg_iters": avg_iters,
            }

            print(f"| Avg {avg_tps:>8.0f} tok/s, Max {max_tps:>8.0f} tok/s")
        else:
            print("| ERROR - no results")

    # 总结
    print("\n" + "=" * 80)
    print("总结 - Qwen3.6-35B-A3B-AWQ Prefill 吞吐量")
    print("=" * 80)
    for label in ["4K", "8K", "16K"]:
        if label in results:
            r = results[label]
            print(f"  {label:>6}: {r['avg_tps']:>10.0f} tok/s "
          f"(TTFT: {r['avg_ms']:>8.1f}ms, Iters: {r.get('avg_iters', 0):.1f})")

    output = {
        "model": MODEL,
        "config": {
            "warmup_runs": WARMUP_RUNS, "measure_runs": MEASURE_RUNS,
            "max_prefill_token_num": 65536,
            "num_tokens_per_iter": 65536,
            "max_prefill_iters": 1,
        },
        "results": results,
        "timestamp": time.time(),
    }

    out_path = Path(__file__).parent / "prefill_benchmark_final_v3.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    tm.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_benchmark())