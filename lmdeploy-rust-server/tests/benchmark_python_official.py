#!/usr/bin/env python3
"""
使用官方 Python TurboMind benchmark 方法
复现历史 42,875 tok/s 的 prefill 性能
"""

import os, sys, time, json
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
WARMUP_RUNS = 2
MEASURE_RUNS = 3
OUTPUT_TOKENS = 512

REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count: int) -> str:
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats

TEST_CONFIGS = {
    "1K": {"context": 1024, "output": 512},
    "4K": {"context": 4096, "output": 512},
    "8K": {"context": 8192, "output": 512},
}


def run_benchmark_with_pipeline():
    """使用官方 pipeline.stream_infer 方法"""
    print("=" * 80)
    print("Python TurboMind Prefill Benchmark (官方方法)")
    print("=" * 80)
    print(f"\nModel: {MODEL}")

    # 创建 pipeline
    print("\n[1/3] 创建 pipeline...", end=" ", flush=True)
    pipe = pipeline(
        MODEL,
        backend_config=TurbomindEngineConfig(
            session_len=16384,
            max_batch_size=1,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            model_format='awq',
            cache_max_entry_count=0.4,
        ),
        log_level='ERROR',
    )
    print("OK")

    results = {}

    for label, config in TEST_CONFIGS.items():
        context_len = config["context"]
        output_len = config["output"]

        print(f"\n[2/3] 测试 {label} context ({context_len} + {output_len})...")
        prompt = gen_prompt(context_len)
        gen_cfg = GenerationConfig(
            max_new_tokens=output_len,
            temperature=0.7,
            do_sample=False,
        )

        # Warmup
        print(f"  Warmup ({WARMUP_RUNS} runs)...", end=" ", flush=True)
        for _ in range(WARMUP_RUNS):
            for _ in pipe.stream_infer(
                [prompt],
                gen_config=gen_cfg,
                do_preprocess=False,
                stream_response=True,
            ):
                pass
        print("OK")

        # 测量
        print(f"  Measure ({MEASURE_RUNS} runs)...", end=" ", flush=True)
        ttfts = []
        total_times = []
        output_counts = []

        for r in range(MEASURE_RUNS):
            start_time = time.perf_counter()
            ttft_time = None
            output_tokens = 0

            for response in pipe.stream_infer(
                [prompt],
                gen_config=gen_cfg,
                do_preprocess=False,
                stream_response=True,
            ):
                if ttft_time is None and response.text:
                    ttft_time = time.perf_counter() - start_time

                if response.token_ids:
                    output_tokens = len(response.token_ids)

            total_time = time.perf_counter() - start_time

            ttfts.append(ttft_time * 1000 if ttft_time else 0)
            total_times.append(total_time * 1000)
            output_counts.append(output_tokens)
            print(f"{ttfts[-1]:.0f}ms ", end="", flush=True)

        # 统计
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_total = sum(total_times) / len(total_times)
        avg_output = sum(output_counts) / len(output_counts)

        prefill_tps = (context_len / avg_ttft * 1000) if avg_ttft > 0 else 0
        decode_tps = (avg_output / ((avg_total - avg_ttft) / 1000)) if (avg_total - avg_ttft) > 0 else 0
        avg_itl = ((avg_total - avg_ttft) / avg_output) if avg_output > 0 else 0

        results[label] = {
            "context_length": context_len,
            "output_tokens": output_len,
            "ttft_ms_avg": avg_ttft,
            "total_time_ms_avg": avg_total,
            "prefill_speed_tps_avg": prefill_tps,
            "decode_speed_tps_avg": decode_tps,
            "avg_itl_ms_avg": avg_itl,
        }

        print(f"\n  TTFT: {avg_ttft:.2f}ms, Prefill: {prefill_tps:.0f} tok/s, Decode: {decode_tps:.1f} tok/s")

    # 总结
    print("\n" + "=" * 80)
    print("总结 - Python TurboMind Prefill 性能")
    print("=" * 80)

    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"  {label:>6}: TTFT={r['ttft_ms_avg']:>7.2f}ms, "
                  f"Prefill={r['prefill_speed_tps_avg']:>8.0f} tok/s, "
                  f"Decode={r['decode_speed_tps_avg']:>5.1f} tok/s")

    # 对比历史数据
    print("\n" + "=" * 80)
    print("与历史数据对比 (BENCHMARK_PYTHON_TM_20260518.json)")
    print("=" * 80)

    historical = {
        "1K": {"ttft": 69.62, "prefill": 14827.2, "decode": 41.2},
        "4K": {"ttft": 122.06, "prefill": 33727.5, "decode": 41.0},
        "8K": {"ttft": 191.37, "prefill": 42875.0, "decode": 40.6},
    }

    print(f"\n{'Context':>8} | {'TTFT (ms)':>12} | {'Prefill (tok/s)':>18} | {'Decode (tok/s)':>15}")
    print("-" * 70)
    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            h = historical[label]
            print(f"{label:>8} | 当前: {r['ttft_ms_avg']:>7.2f} | 当前: {r['prefill_speed_tps_avg']:>10.0f} | 当前: {r['decode_speed_tps_avg']:>7.1f}")
            print(f"         | 历史: {h['ttft']:>7.2f} | 历史: {h['prefill']:>10.0f} | 历史: {h['decode']:>7.1f}")
            ttft_ratio = r['ttft_ms_avg'] / h['ttft'] if h['ttft'] > 0 else 0
            prefill_ratio = h['prefill'] / r['prefill_speed_tps_avg'] if r['prefill_speed_tps_avg'] > 0 else 0
            print(f"         | 差异: {ttft_ratio:>6.1f}x | 差异: {prefill_ratio:>6.1f}x")
            print()

    output = {
        "engine": "LMDeploy Python TurboMind",
        "model": MODEL,
        "date": time.time(),
        "results": results,
        "historical": historical,
    }

    out_path = Path(__file__).parent / "benchmark_python_official.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    run_benchmark_with_pipeline()