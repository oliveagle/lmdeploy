#!/usr/bin/env python3
"""
正确测量 TurboMind C++ Prefill 性能

关键修正：
1. 引擎只创建一次
2. Warmup 后再测量
3. 排除 tokenization 时间
4. 纯测量 C++ prefill throughput
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

# 配置
MODEL = os.environ.get(
    "MODEL_PATH",
    "/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
)
WARMUP_RUNS = 2
MEASURE_RUNS = 5

# 测试 prompts (纯重复文本，避免其他开销)
TEST_PROMPTS = {
    "256": "The quick brown fox jumps over the lazy dog. " * 40,
    "512": "The quick brown fox jumps over the lazy dog. " * 80,
    "1024": "The quick brown fox jumps over the lazy dog. " * 160,
    "2048": "The quick brown fox jumps over the lazy dog. " * 320,
    "4096": "The quick brown fox jumps over the lazy dog. " * 640,
    "8192": "The quick brown fox jumps over the lazy dog. " * 1280,
}


async def measure_prefill_single(
    inst,
    session_id: int,
    input_ids,
    gen_cfg,
) -> float | None:
    """测量单次 prefill TTFT (ms)"""
    start = time.perf_counter()
    first_token_ms = None

    async for out in inst.async_stream_infer(
        session_id=session_id,
        input_ids=input_ids,
        gen_config=gen_cfg,
        sequence_start=True,
        sequence_end=True,
    ):
        if out.status.value in (1, 2):  # 有输出
            first_token_ms = (time.perf_counter() - start) * 1000
            break

    return first_token_ms


async def measure_prefill_performance(
    inst,
    tok: Tokenizer,
    prompt: str,
    runs: int = MEASURE_RUNS,
) -> Dict[str, Any]:
    """测量 prefill 性能 - 引擎已创建"""

    # Tokenization
    input_ids = tok.encode(prompt)
    actual_ctx = len(input_ids)
    gen_cfg = GenerationConfig(max_new_tokens=1, temperature=0.7)

    run_times = []

    for r in range(runs):
        first_token_ms = await measure_prefill_single(
            inst, r + 1000, input_ids, gen_cfg
        )
        if first_token_ms:
            run_times.append(first_token_ms)

    # 统计
    avg_ms = sum(run_times) / len(run_times)
    min_ms = min(run_times)
    max_ms = max(run_times)

    # 计算 throughput
    prefill_tps_avg = (actual_ctx / avg_ms * 1000) if avg_ms > 0 else 0
    prefill_tps_max = (actual_ctx / min_ms * 1000) if min_ms > 0 else 0

    return {
        "ctx_tokens": actual_ctx,
        "runs_ms": run_times,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "prefill_tps_avg": prefill_tps_avg,
        "prefill_tps_max": prefill_tps_max,
    }


async def warmup(inst, tok, warmup_prompt):
    """Warmup"""
    warmup_ids = tok.encode(warmup_prompt)
    gen_cfg = GenerationConfig(max_new_tokens=1)
    for i in range(WARMUP_RUNS):
        await measure_prefill_single(inst, i, warmup_ids, gen_cfg)


async def main():
    print("=" * 80)
    print("TurboMind C++ Prefill 性能测试 (正确版本)")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    print(f"Warmup: {WARMUP_RUNS}, Measure: {MEASURE_RUNS}")
    print(f"Output tokens: 1 (只测量 prefill)")

    # 创建引擎 (只创建一次!)
    print("\n[1/3] 创建 TurboMind 引擎...", end=" ", flush=True)
    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=8192 * 2,
            max_batch_size=32,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            dtype="bfloat16",  # Model is bfloat16
            max_prefill_token_num=8192,
        ),
        trust_remote_code=True,
    )
    tok = Tokenizer(MODEL, trust_remote_code=True)
    inst = tm.create_instance()
    print("OK")

    # Warmup
    print(f"[2/3] Warmup ({WARMUP_RUNS} runs)...", end=" ", flush=True)
    await warmup(inst, tok, TEST_PROMPTS["512"])
    print("OK")

    # 测量
    print(f"[3/3] 测量 prefill 性能...")
    results = {}

    print(f"\n{'Context':>8} | {'Avg TTFT':>10} | {'Min TTFT':>10} | "
          f"{'Avg TPS':>12} | {'Max TPS':>12} | {'Runs':>5}")
    print("-" * 80)

    for ctx_label, prompt in TEST_PROMPTS.items():
        result = await measure_prefill_performance(inst, tok, prompt, MEASURE_RUNS)
        results[ctx_label] = result

        print(f"{result['ctx_tokens']:>8} | "
              f"{result['avg_ms']:>10.2f} | "
              f"{result['min_ms']:>10.2f} | "
              f"{result['prefill_tps_avg']:>12.0f} | "
              f"{result['prefill_tps_max']:>12.0f} | "
              f"{MEASURE_RUNS:>5}")

    # 总结
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    for ctx_label in ["256", "512", "1024", "2048", "4096", "8192"]:
        if ctx_label in results:
            r = results[ctx_label]
            print(f"\n  {ctx_label} tokens:")
            print(f"    Avg Prefill: {r['prefill_tps_avg']:.0f} tok/s (TTFT: {r['avg_ms']:.2f}ms)")
            print(f"    Max Prefill: {r['prefill_tps_max']:.0f} tok/s (TTFT: {r['min_ms']:.2f}ms)")

    # 保存结果
    output = {
        "model": MODEL,
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "measure_runs": MEASURE_RUNS,
            "output_tokens": 1,
        },
        "results": results,
        "timestamp": time.time(),
    }

    out_path = Path(__file__).parent / "prefill_benchmark_correct.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    tm.close()


if __name__ == "__main__":
    asyncio.run(main())