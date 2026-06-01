#!/usr/bin/env python3
"""
LMDeploy TurboMind Prefill Benchmark - 4k, 8k, 16k
基于 benchmark_prefill_verification.py，修改测试场景为 4k, 8k, 16k
"""
import asyncio
import json
import time
import os
import gc
from datetime import datetime

from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig
import numpy as np

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
MIN_STABLE_RUNS = 3
OUTLIER_THRESHOLD = 2.0


def generate_prompt(tokenizer, target_len):
    """生成指定 token 长度的 prompt"""
    base_text = "Hello, how are you today? This is a test message for benchmarking. "
    base_tokens = tokenizer.encode(base_text)
    repeat_times = (target_len // len(base_tokens)) + 1
    input_ids = (base_tokens * repeat_times)[:target_len]
    return tokenizer.decode(input_ids), input_ids


async def run_single_benchmark(model_inst, input_ids, output_len, session_id):
    """运行单次基准测试，返回详细指标"""
    params = GenerationConfig(
        max_new_tokens=output_len,
        ignore_eos=True,
        min_new_tokens=output_len,
    )

    start = time.perf_counter()
    ttft = None
    token_times = []
    token_count = 0

    try:
        gen = model_inst.async_stream_infer(
            session_id,
            input_ids=input_ids,
            gen_config=params,
            sequence_start=True,
            sequence_end=True,
            stream_output=True,
        )
        async for outputs in gen:
            now = time.perf_counter()

            if token_count == 0:
                ttft = now - start

            token_count += len(outputs.token_ids)
            token_times.append(now)

        await gen.aclose()

        total_time = time.perf_counter() - start

        # 计算 TPOT (Time Per Output Token)
        tpots = []
        if len(token_times) > 1:
            for i in range(1, len(token_times)):
                tpots.append(token_times[i] - token_times[i-1])

        return {
            "ttft": ttft,
            "total_time": total_time,
            "token_count": token_count,
            "tpots": tpots,
            "success": True,
        }

    except Exception as e:
        return {
            "ttft": None,
            "total_time": None,
            "token_count": 0,
            "tpots": [],
            "success": False,
            "error": str(e),
        }


def remove_outliers(data, threshold=OUTLIER_THRESHOLD):
    """移除异常值"""
    if not data:
        return []

    median = np.median(data)
    valid_data = [x for x in data if x < median * threshold and x > median / threshold]
    return valid_data


async def benchmark_scenario(tm_model, tokenizer, scenario):
    """基准测试单个场景"""
    input_len = scenario["input_len"]
    output_len = scenario["output_len"]

    _, input_ids = generate_prompt(tokenizer, input_len)
    actual_input_len = len(input_ids)

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario['name']}")
    print(f"Input length: {actual_input_len}, Output length: {output_len}")
    print(f"{'='*70}")

    results = []
    ttfts = []
    all_tpots = []
    total_times = []

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} requests...")
    model_inst = tm_model.create_instance()
    for i in range(WARMUP_REQUESTS):
        await run_single_benchmark(model_inst, [1, 2, 3, 4, 5], 10, i)
        await asyncio.sleep(0.1)

    # 实际基准测试
    print(f"Benchmark: {NUM_REQUESTS} requests...")
    for i in range(NUM_REQUESTS):
        # 强制 GC 减少内存影响
        gc.collect()

        result = await run_single_benchmark(
            model_inst, input_ids, output_len, WARMUP_REQUESTS + i)

        if result["success"] and result["ttft"] is not None:
            results.append(result)
            ttfts.append(result["ttft"])
            total_times.append(result["total_time"])
            all_tpots.extend(result["tpots"])

            ttft_ms = result["ttft"] * 1000
            total_ms = result["total_time"] * 1000
            prefill_tps = actual_input_len / result["ttft"]
            decode_tps = (result["token_count"] - 1) / (result["total_time"] - result["ttft"]) if result["total_time"] > result["ttft"] else 0
            print(f"  Run {i+1:2d}: TTFT={ttft_ms:7.1f}ms, Total={total_ms:7.1f}ms, Prefill={prefill_tps:7.1f} tok/s, Decode={decode_tps:6.1f} tok/s")
        else:
            print(f"  Run {i+1:2d}: FAILED - {result.get('error', 'unknown')}")

        await asyncio.sleep(0.1)  # 避免请求过于密集

    if not results:
        print("No valid results!")
        return None

    # 移除异常值
    ttfts_valid = remove_outliers(ttfts)
    total_times_valid = remove_outliers(total_times)
    tpots_valid = remove_outliers(all_tpots)

    if len(ttfts_valid) < MIN_STABLE_RUNS:
        print(f"Warning: Only {len(ttfts_valid)} valid runs (need {MIN_STABLE_RUNS}), using all runs")
        ttfts_valid = ttfts
        total_times_valid = total_times
        tpots_valid = all_tpots

    # 统计指标
    ttfts_arr = np.array(ttfts_valid)
    total_times_arr = np.array(total_times_valid)
    tpots_arr = np.array(tpots_valid)

    avg_ttft = np.mean(ttfts_arr)
    median_ttft = np.median(ttfts_arr)
    median_total = np.median(total_times_arr)
    median_tpot = np.median(tpots_arr) if len(tpots_arr) > 0 else 0

    # 使用 median 计算吞吐量，更稳定
    prefill_tps_median = actual_input_len / median_ttft if median_ttft > 0 else 0
    prefill_tps_avg = actual_input_len / avg_ttft if avg_ttft > 0 else 0

    # Decode 吞吐量 = (输出 tokens - 1) / (总时间 - TTFT)
    decode_time = median_total - median_ttft
    decode_throughput = (output_len - 1) / decode_time if decode_time > 0 else 0
    overall_throughput = output_len / median_total if median_total > 0 else 0

    stats = {
        "scenario": scenario["name"],
        "input_len": actual_input_len,
        "output_len": output_len,
        "total_runs": NUM_REQUESTS,
        "valid_runs": len(ttfts_valid),
        "warmup_runs": WARMUP_REQUESTS,
        "ttft_ms_avg": avg_ttft * 1000,
        "ttft_ms_median": median_ttft * 1000,
        "tpot_ms_median": median_tpot * 1000 if median_tpot > 0 else 0,
        "prefill_throughput_tps_median": prefill_tps_median,
        "prefill_throughput_tps_avg": prefill_tps_avg,
        "decode_tps": decode_throughput,
        "overall_tps": overall_throughput,
        "decode_time_ms": decode_time * 1000,
        "total_time_ms_median": median_total * 1000,
    }

    print(f"\nResults (after outlier removal):")
    print(f"  Valid runs: {stats['valid_runs']}/{stats['total_runs']}")
    print(f"  TTFT: median={stats['ttft_ms_median']:.1f}ms, mean={stats['ttft_ms_avg']:.1f}ms")
    print(f"  Prefill throughput: median={stats['prefill_throughput_tps_median']:.1f} tok/s")
    print(f"  Decode throughput: {stats['decode_tps']:.1f} tok/s")
    print(f"  Overall throughput: {stats['overall_tps']:.1f} tok/s")

    return stats


async def main():
    print(f"{'='*70}")
    print(f"LMDeploy TurboMind Prefill Benchmark (4k, 8k, 16k)")
    print(f"Model: {MODEL_PATH}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    print("Initializing TurboMind engine...")
    tm_model = TurboMind.from_pretrained(MODEL_PATH)

    all_results = []

    for scenario in SCENARIOS:
        result = await benchmark_scenario(tm_model, tokenizer, scenario)
        if result:
            all_results.append(result)
        await asyncio.sleep(1)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_json = f"benchmark_python_turbomind_4k_8k_16k_{timestamp}.json"
    output_file_md = f"benchmark_python_turbomind_4k_8k_16k_{timestamp}.md"

    output = {
        "engine": "LMDeploy TurboMind",
        "model": MODEL_PATH,
        "backend": "turbomind",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "Single request, serial execution, TTFT-based prefill calculation",
        "results": all_results,
    }

    # 保存 JSON
    with open(output_file_json, "w") as f:
        json.dump(output, f, indent=2)

    # 保存 Markdown 报告
    with open(output_file_md, "w") as f:
        f.write(f"# Python + TurboMind 性能基准报告 (4k, 8k, 16k)\n\n")
        f.write(f"- **测试日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **模型**: {MODEL_PATH}\n")
        f.write(f"- **后端**: TurboMind C++\n")
        f.write(f"- **测试方法**: 单请求串行，TTFT 计算 Prefill 吞吐量\n\n")

        f.write(f"## 测试结果\n\n")
        f.write(f"| 输入长度 | TTFT median (ms) | Prefill median (tok/s) | Decode (tok/s) | Overall (tok/s) | Valid runs |\n")
        f.write(f"|----------|------------------|------------------------|----------------|-----------------|------------|\n")
        for r in all_results:
            f.write(f"| {r['input_len']:8d} | {r['ttft_ms_median']:16.1f} | {r['prefill_throughput_tps_median']:22.1f} | {r['decode_tps']:14.1f} | {r['overall_tps']:15.1f} | {r['valid_runs']:10d} |\n")

        f.write(f"\n## 详细数据\n\n")
        f.write(f"```json\n{json.dumps(all_results, indent=2)}\n```\n")

    print(f"\n{'='*70}")
    print(f"Results saved:")
    print(f"  - {output_file_json}")
    print(f"  - {output_file_md}")
    print(f"{'='*70}")

    # 打印摘要表格
    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Input Len':>10} | {'TTFT (ms)':>12} | {'Prefill (tok/s)':>18} | {'Decode (tok/s)':>16}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['input_len']:10d} | {r['ttft_ms_median']:12.1f} | {r['prefill_throughput_tps_median']:18.1f} | {r['decode_tps']:16.1f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
