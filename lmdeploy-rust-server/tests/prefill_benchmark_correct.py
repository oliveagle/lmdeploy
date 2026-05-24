#!/usr/bin/env python3
"""
测量纯 Prefill 时间（不包含 decode）

使用 output_embeddings=True 或只调用一次 prefill
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path
from collections import defaultdict

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

# 配置
MODEL = os.environ.get(
    "MODEL_PATH",
    "/mnt/data/models/lmdeploy_models/Qwen3.5-9B",
)
WARMUP_RUNS = 2
MEASURE_RUNS = 3

# 测试 prompts
TEST_PROMPTS = {
    "256": "The quick brown fox jumps over the lazy dog. " * 40,
    "512": "The quick brown fox jumps over the lazy dog. " * 80,
    "1024": "The quick brown fox jumps over the lazy dog. " * 160,
    "2048": "The quick brown fox jumps over the lazy dog. " * 320,
    "4096": "The quick brown fox jumps over the lazy dog. " * 640,
}


async def run_prefill_benchmark():
    """运行完整的 prefill benchmark"""
    print("=" * 80)
    print("TurboMind C++ Prefill 性能测试")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    print(f"Warmup: {WARMUP_RUNS}, Measure: {MEASURE_RUNS}")

    # 创建引擎
    print("\n[1/3] 创建引擎...", end=" ", flush=True)
    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=8192 * 2,
            max_batch_size=32,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            max_prefill_token_num=16384,
        ),
        trust_remote_code=True,
    )
    tok = Tokenizer(MODEL, trust_remote_code=True)
    inst = tm.create_instance()
    print("OK")

    # Warmup
    print(f"[2/3] Warmup...", end=" ", flush=True)
    warmup_ids = tok.encode(TEST_PROMPTS["512"])
    for i in range(WARMUP_RUNS):
        for _ in inst.async_stream_infer(
            session_id=i,
            input_ids=warmup_ids,
            gen_config=GenerationConfig(max_new_tokens=1),
            sequence_start=True,
            sequence_end=True,
        ):
            pass
    print("OK")

    # 测量
    print(f"[3/3] 测量预填充性能...")
    results = {}
    gen_cfg = GenerationConfig(max_new_tokens=1, temperature=0.7)

    print(f"\n{'Context':>8} | {'Avg TTFT':>10} | {'Min TTFT':>10} | "
          f"{'Avg TPS':>12} | {'Max TPS':>12} | {'Runs':>5}")
    print("-" * 80)

    for ctx_label, prompt in TEST_PROMPTS.items():
        input_ids = tok.encode(prompt)
        actual_ctx = len(input_ids)
        run_times = []

        for r in range(MEASURE_RUNS):
            start = time.perf_counter()
            first_token_ms = None

            for out in inst.async_stream_infer(
                session_id=r + 1000 + hash(ctx_label) % 100,
                input_ids=input_ids,
                gen_config=gen_cfg,
                sequence_start=True,
                sequence_end=True,
            ):
                if out.status.value in (1, 2):
                    first_token_ms = (time.perf_counter() - start) * 1000
                    break

            if first_token_ms:
                run_times.append(first_token_ms)

        if run_times:
            avg_ms = sum(run_times) / len(run_times)
            min_ms = min(run_times)
            max_ms = max(run_times)
            avg_tps = (actual_ctx / avg_ms * 1000) if avg_ms > 0 else 0
            max_tps = (actual_ctx / min_ms * 1000) if min_ms > 0 else 0

            results[ctx_label] = {
                "ctx_tokens": actual_ctx,
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "avg_tps": avg_tps,
                "max_tps": max_tps,
            }

            print(f"{actual_ctx:>8} | {avg_ms:>10.2f} | {min_ms:>10.2f} | "
                  f"{avg_tps:>12.0f} | {max_tps:>12.0f} | {len(run_times):>5}")
        else:
            print(f"{actual_ctx:>8} | ERROR - no results")

    # 打印总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    for ctx_label in ["256", "512", "1024", "2048", "4096"]:
        if ctx_label in results:
            r = results[ctx_label]
            print(f"  {ctx_label} tokens: Avg {r['avg_tps']:.0f} tok/s "
                  f"(TTFT: {r['avg_ms']:.2f}ms)")

    # 保存结果
    out_path = Path(__file__).parent / "prefill_benchmark_final.json"
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "results": results}, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    tm.close()


if __name__ == "__main__":
    asyncio.run(run_prefill_benchmark())