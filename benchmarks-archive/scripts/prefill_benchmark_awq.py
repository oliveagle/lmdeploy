#!/usr/bin/env python3
"""
TurboMind C++ Prefill 性能测试 - Qwen3.6-35B-A3B-AWQ
测试: 4K, 8K, 16K, 32K tokens
"""

import os, sys, asyncio, time, json
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
WARMUP_RUNS = 2
MEASURE_RUNS = 5  # 多测几轮取平均

REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count: int) -> str:
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats

TEST_PROMPTS = {
    "4K": 4000,
    "8K": 8000,
    "16K": 16000,
    "32K": 32000,
}


async def run_benchmark():
    print("=" * 80)
    print("TurboMind C++ Prefill 性能测试 - Qwen3.6-35B-A3B-AWQ")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    print(f"Warmup: {WARMUP_RUNS}, Measure: {MEASURE_RUNS}")

    print("\n[1/3] 创建引擎...", end=" ", flush=True)
    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=32768,
            max_batch_size=1,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            max_prefill_token_num=32768,
            model_format='awq',
            cache_max_entry_count=0.4,  # 降低 cache 使用量给 32K 预留空间
        ),
    )
    tok = Tokenizer(MODEL)
    inst = tm.create_instance()
    print("OK")

    # Warmup
    print(f"[2/3] Warmup...", end=" ", flush=True)
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

    print(f"\n{'Label':>6} | {'Tokens':>8} | {'Avg TTFT':>10} | {'Min TTFT':>10} | "
          f"{'Avg TPS':>12} | {'Max TPS':>12}")
    print("-" * 80)

    for label, target_tokens in TEST_PROMPTS.items():
        prompt = gen_prompt(target_tokens)
        input_ids = tok.encode(prompt)
        actual_tokens = len(input_ids)
        run_times = []

        print(f"{label:>6} ({actual_tokens:>6} tok)... ", end="", flush=True)

        for r in range(MEASURE_RUNS):
            session_id = r + 2000 + hash(label) % 1000
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

    print("\n" + "=" * 80)
    print("总结 - Qwen3.6-35B-A3B-AWQ Prefill 性能")
    print("=" * 80)
    for label in ["4K", "8K", "16K", "32K"]:
        if label in results:
            r = results[label]
            print(f"  {label:>6}: {r['avg_tps']:>10.0f} tok/s (TTFT: {r['avg_ms']:>8.1f}ms)")

    output = {
        "model": MODEL,
        "config": {"warmup_runs": WARMUP_RUNS, "measure_runs": MEASURE_RUNS},
        "results": results,
        "timestamp": time.time(),
    }

    out_path = Path(__file__).parent / "prefill_benchmark_awq_35b_final.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    tm.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())