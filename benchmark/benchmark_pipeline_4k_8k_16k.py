#!/usr/bin/env python3
"""
LMDeploy Pipeline Benchmark - 4k, 8k, 16k
使用 pipeline API 进行性能测试，避免内部 API 兼容性问题
"""
import time
import os
import gc
from datetime import datetime
import json

from lmdeploy import pipeline

# 模型路径
MODEL_PATH = os.getenv("LMDEPLOY_BENCH_MODEL",
                       "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ")

# 测试配置 - 4k, 8k, 16k
SCENARIOS = [
    {"name": "4k", "input_len": 4096, "output_len": 512},
    {"name": "8k", "input_len": 8192, "output_len": 512},
    {"name": "16k", "input_len": 16384, "output_len": 512},
]

NUM_REQUESTS = 5
WARMUP_REQUESTS = 2

REPEAT_TEXT = "Hello, how are you today? This is a test message for benchmarking. "


def generate_prompt(target_len):
    """生成指定 token 长度的 prompt"""
    repeat_times = (target_len // len(REPEAT_TEXT.split())) + 1
    words = REPEAT_TEXT.split() * repeat_times
    return " ".join(words[:target_len])


def benchmark_scenario(pipe, scenario):
    """基准测试单个场景"""
    input_len = scenario["input_len"]
    output_len = scenario["output_len"]

    # Generate prompt
    prompt = generate_prompt(input_len)
    actual_words = len(prompt.split())

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario['name']}")
    print(f"Target Input: {input_len}, Output: {output_len}")
    print(f"Actual words: {actual_words}")
    print(f"{'='*70}")

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} runs...")
    for i in range(WARMUP_REQUESTS):
        for _ in pipe.stream_infer("Hello, this is a warmup test.", max_tokens=10):
            pass
        print(f"  warmup {i+1}... OK")

    # Benchmark
    print(f"Benchmark: {NUM_REQUESTS} runs...")
    ttfts = []
    total_times = []

    for i in range(NUM_REQUESTS):
        gc.collect()

        start = time.perf_counter()
        first_token = None
        token_count = 0

        for output in pipe.stream_infer(
            prompt,
            max_tokens=output_len,
            ignore_eos=True,
            min_tokens=output_len,
        ):
            if first_token is None:
                first_token = time.perf_counter() - start
            token_count += 1

        total = time.perf_counter() - start

        ttfts.append(first_token)
        total_times.append(total)

        prefill_tps = input_len / first_token if first_token > 0 else 0
        decode_tps = (output_len - 1) / (total - first_token) if total > first_token else 0
        print(f"  Run {i+1:2d}: TTFT={first_token*1000:7.1f}ms, Total={total*1000:7.1f}ms, Prefill={prefill_tps:7.1f} tok/s, Decode={decode_tps:6.1f} tok/s")

    # 统计
    import numpy as np
    ttfts_arr = np.array(ttfts)
    total_times_arr = np.array(total_times)

    median_ttft = np.median(ttfts_arr)
    avg_ttft = np.mean(ttfts_arr)
    median_total = np.median(total_times_arr)

    prefill_tps_median = input_len / median_ttft if median_ttft > 0 else 0
    prefill_tps_avg = input_len / avg_ttft if avg_ttft > 0 else 0

    decode_time = median_total - median_ttft
    decode_tps = (output_len - 1) / decode_time if decode_time > 0 else 0
    overall_tps = output_len / median_total if median_total > 0 else 0

    print(f"\nResults:")
    print(f"  TTFT: median={median_ttft*1000:.1f}ms, mean={avg_ttft*1000:.1f}ms")
    print(f"  Prefill: median={prefill_tps_median:.1f} tok/s")
    print(f"  Decode: {decode_tps:.1f} tok/s")
    print(f"  Overall: {overall_tps:.1f} tok/s")

    return {
        "scenario": scenario["name"],
        "input_len": input_len,
        "output_len": output_len,
        "iterations": NUM_REQUESTS,
        "ttft_ms_median": median_ttft * 1000,
        "ttft_ms_avg": avg_ttft * 1000,
        "prefill_tps_median": prefill_tps_median,
        "prefill_tps_avg": prefill_tps_avg,
        "decode_tps": decode_tps,
        "overall_tps": overall_tps,
        "total_time_ms_median": median_total * 1000,
    }


def main():
    print(f"{'='*70}")
    print(f"LMDeploy Pipeline Benchmark (4k, 8k, 16k)")
    print(f"Model: {MODEL_PATH}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    print("\nInitializing pipeline...")
    pipe = pipeline(MODEL_PATH)

    all_results = []

    for scenario in SCENARIOS:
        result = benchmark_scenario(pipe, scenario)
        if result:
            all_results.append(result)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_pipeline_4k_8k_16k_{timestamp}.json"

    output = {
        "engine": "LMDeploy Pipeline",
        "model": MODEL_PATH,
        "backend": "turbomind",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "Pipeline API, serial execution",
        "warmup_runs": WARMUP_REQUESTS,
        "measure_runs": NUM_REQUESTS,
        "output_length": 512,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # 打印摘要
    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Input Len':>10} | {'TTFT (ms)':>12} | {'Prefill (tok/s)':>18} | {'Decode (tok/s)':>16}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['input_len']:10d} | {r['ttft_ms_median']:12.1f} | {r['prefill_tps_median']:18.1f} | {r['decode_tps']:16.1f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
