#!/usr/bin/env python3
"""
LMDeploy TurboMind Complete Performance Benchmark
完整的 TurboMind 性能基准测试，包括 Prefill 和 Decode 阶段。

测试方法:
1. 使用 TurboMind 引擎直接测试
2. 排除预热数据，取稳定值
3. 分别测试 Prefill (TTFT) 和 Decode (TPOT) 性能
4. 多次运行取中位数/平均，移除异常值

模型: Qwen3.6-35B-A3B-AWQ (默认，可通过环境变量覆盖)
"""
import asyncio
import json
import time
import os
import gc
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig

# 模型路径 (可通过环境变量覆盖)
MODEL_PATH = os.getenv("LMDEPLOY_BENCH_MODEL",
                       "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ")

# 测试配置
NUM_REQUESTS = 10
WARMUP_REQUESTS = 3
MIN_STABLE_RUNS = 5
OUTLIER_THRESHOLD = 2.0

# Prefill 测试场景 (单 token 输出，测量 TTFT)
PREFILL_SCENARIOS = [
    {"name": "prefill_128", "input_len": 128, "output_len": 1},
    {"name": "prefill_256", "input_len": 256, "output_len": 1},
    {"name": "prefill_512", "input_len": 512, "output_len": 1},
    {"name": "prefill_1024", "input_len": 1024, "output_len": 1},
    {"name": "prefill_2048", "input_len": 2048, "output_len": 1},
    {"name": "prefill_4096", "input_len": 4096, "output_len": 1},
]

# Decode 测试场景 (长输出，测量 TPOT 和吞吐量)
DECODE_SCENARIOS = [
    {"name": "decode_32", "input_len": 32, "output_len": 32},
    {"name": "decode_64", "input_len": 32, "output_len": 64},
    {"name": "decode_128", "input_len": 32, "output_len": 128},
    {"name": "decode_256", "input_len": 32, "output_len": 256},
    {"name": "decode_512", "input_len": 32, "output_len": 512},
]


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


async def benchmark_prefill(tm_model, tokenizer, scenario):
    """基准测试 Prefill 阶段"""
    input_len = scenario["input_len"]
    output_len = scenario["output_len"]

    _, input_ids = generate_prompt(tokenizer, input_len)
    actual_input_len = len(input_ids)

    print(f"\n{'='*70}")
    print(f"Prefill Scenario: {scenario['name']}")
    print(f"Input: {actual_input_len}, Output: {output_len}")
    print(f"{'='*70}")

    model_inst = tm_model.create_instance()

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} requests...")
    for i in range(WARMUP_REQUESTS):
        await run_single_benchmark(model_inst, [1, 2, 3, 4, 5], 10, i)
        await asyncio.sleep(0.1)

    # Benchmark
    print(f"Benchmark: {NUM_REQUESTS} requests...")
    results = []
    ttfts = []

    for i in range(NUM_REQUESTS):
        gc.collect()

        result = await run_single_benchmark(
            model_inst, input_ids, output_len, WARMUP_REQUESTS + i)

        if result["success"] and result["ttft"] is not None:
            results.append(result)
            ttfts.append(result["ttft"])
            ttft_ms = result["ttft"] * 1000
            prefill_tps = actual_input_len / result["ttft"]
            print(f"  Run {i+1:2d}: TTFT={ttft_ms:7.1f}ms, Prefill={prefill_tps:7.1f} tok/s")
        else:
            print(f"  Run {i+1:2d}: FAILED")

        await asyncio.sleep(0.1)

    if not results:
        return None

    # 移除异常值
    ttfts_valid = remove_outliers(ttfts)
    if len(ttfts_valid) < MIN_STABLE_RUNS:
        ttfts_valid = ttfts

    ttfts_arr = np.array(ttfts_valid)
    median_ttft = np.median(ttfts_arr)
    avg_ttft = np.mean(ttfts_arr)

    return {
        "scenario": scenario["name"],
        "input_len": actual_input_len,
        "output_len": output_len,
        "valid_runs": len(ttfts_valid),
        "ttft_ms_median": median_ttft * 1000,
        "ttft_ms_avg": avg_ttft * 1000,
        "prefill_throughput_tps_median": actual_input_len / median_ttft if median_ttft > 0 else 0,
    }


async def benchmark_decode(tm_model, tokenizer, scenario):
    """基准测试 Decode 阶段"""
    input_len = scenario["input_len"]
    output_len = scenario["output_len"]

    _, input_ids = generate_prompt(tokenizer, input_len)
    actual_input_len = len(input_ids)

    print(f"\n{'='*70}")
    print(f"Decode Scenario: {scenario['name']}")
    print(f"Input: {actual_input_len}, Output: {output_len}")
    print(f"{'='*70}")

    model_inst = tm_model.create_instance()

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} requests...")
    for i in range(WARMUP_REQUESTS):
        await run_single_benchmark(model_inst, [1, 2, 3, 4, 5], 10, i)
        await asyncio.sleep(0.1)

    # Benchmark
    print(f"Benchmark: {NUM_REQUESTS} requests...")
    results = []
    ttfts = []
    all_tpots = []
    total_times = []

    for i in range(NUM_REQUESTS):
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
            avg_tpot = np.mean(result["tpots"]) * 1000 if result["tpots"] else 0
            decode_tps = (result["token_count"] - 1) / (result["total_time"] - result["ttft"]) if result["total_time"] > result["ttft"] else 0
            print(f"  Run {i+1:2d}: TTFT={ttft_ms:7.1f}ms, Total={total_ms:7.1f}ms, Decode={decode_tps:6.1f} tok/s")
        else:
            print(f"  Run {i+1:2d}: FAILED")

        await asyncio.sleep(0.1)

    if not results:
        return None

    # 移除异常值
    ttfts_valid = remove_outliers(ttfts)
    total_times_valid = remove_outliers(total_times)
    tpots_valid = remove_outliers(all_tpots)

    if len(ttfts_valid) < MIN_STABLE_RUNS:
        ttfts_valid = ttfts
        total_times_valid = total_times
        tpots_valid = all_tpots

    ttfts_arr = np.array(ttfts_valid)
    total_times_arr = np.array(total_times_valid)
    tpots_arr = np.array(tpots_valid)

    median_ttft = np.median(ttfts_arr)
    median_total = np.median(total_times_arr)
    median_tpot = np.median(tpots_arr)

    # Decode 吞吐量 = (输出 tokens - 1) / (总时间 - TTFT)
    decode_time = median_total - median_ttft
    decode_throughput = (output_len - 1) / decode_time if decode_time > 0 else 0

    return {
        "scenario": scenario["name"],
        "input_len": actual_input_len,
        "output_len": output_len,
        "valid_runs": len(ttfts_valid),
        "ttft_ms_median": median_ttft * 1000,
        "total_time_ms_median": median_total * 1000,
        "tpot_ms_median": median_tpot * 1000,
        "decode_time_ms": decode_time * 1000,
        "decode_throughput_tps": decode_throughput,
        "overall_throughput_tps": output_len / median_total if median_total > 0 else 0,
    }


async def main():
    print(f"{'='*70}")
    print(f"LMDeploy TurboMind Complete Performance Benchmark")
    print(f"Model: {MODEL_PATH}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    print("Initializing TurboMind engine...")
    tm_model = TurboMind.from_pretrained(MODEL_PATH)

    all_results = {
        "prefill": [],
        "decode": [],
    }

    # Prefill 基准测试
    print(f"\n{'#'*70}")
    print(f"# Prefill Performance Benchmark")
    print(f"{'#'*70}")

    for scenario in PREFILL_SCENARIOS:
        result = await benchmark_prefill(tm_model, tokenizer, scenario)
        if result:
            all_results["prefill"].append(result)
        await asyncio.sleep(1)

    # Decode 基准测试
    print(f"\n{'#'*70}")
    print(f"# Decode Performance Benchmark")
    print(f"{'#'*70}")

    for scenario in DECODE_SCENARIOS:
        result = await benchmark_decode(tm_model, tokenizer, scenario)
        if result:
            all_results["decode"].append(result)
        await asyncio.sleep(1)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_json = f"benchmark_turbomind_complete_{timestamp}.json"
    output_file_md = f"benchmark_turbomind_complete_{timestamp}.md"

    output = {
        "engine": "LMDeploy TurboMind",
        "model": MODEL_PATH,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "Serial execution, warmup excluded, outlier removal",
        "results": all_results,
    }

    with open(output_file_json, "w") as f:
        json.dump(output, f, indent=2)

    # Markdown 报告
    with open(output_file_md, "w") as f:
        f.write(f"# TurboMind 完整性能基准报告\n\n")
        f.write(f"- **测试日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **模型**: {MODEL_PATH}\n")
        f.write(f"- **后端**: TurboMind C++\n")
        f.write(f"- **方法**: 串行执行，预热排除，异常值移除\n\n")

        f.write(f"## Prefill 性能\n\n")
        f.write(f"| 输入长度 | TTFT median (ms) | Prefill 吞吐量 (tok/s) |\n")
        f.write(f"|----------|------------------|--------------------------|\n")
        for r in all_results["prefill"]:
            f.write(f"| {r['input_len']:8d} | {r['ttft_ms_median']:16.1f} | {r['prefill_throughput_tps_median']:24.1f} |\n")

        f.write(f"\n## Decode 性能\n\n")
        f.write(f"| 输出长度 | TTFT (ms) | TPOT (ms) | Decode 吞吐量 (tok/s) | 整体吞吐量 (tok/s) |\n")
        f.write(f"|----------|-----------|-----------|------------------------|------------------------|\n")
        for r in all_results["decode"]:
            f.write(f"| {r['output_len']:8d} | {r['ttft_ms_median']:9.1f} | {r['tpot_ms_median']:9.1f} | {r['decode_throughput_tps']:22.1f} | {r['overall_throughput_tps']:22.1f} |\n")

    print(f"\n{'='*70}")
    print(f"Results saved:")
    print(f"  - {output_file_json}")
    print(f"  - {output_file_md}")
    print(f"{'='*70}")

    # 打印摘要
    print(f"\n{'='*80}")
    print(f"{'PREFILL SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Input Len':>10} | {'TTFT (ms)':>12} | {'Prefill (tok/s)':>18}")
    print(f"{'-'*80}")
    for r in all_results["prefill"]:
        print(f"{r['input_len']:10d} | {r['ttft_ms_median']:12.1f} | {r['prefill_throughput_tps_median']:18.1f}")

    print(f"\n{'='*80}")
    print(f"{'DECODE SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Output Len':>10} | {'TTFT (ms)':>12} | {'TPOT (ms)':>12} | {'Decode (tok/s)':>18}")
    print(f"{'-'*80}")
    for r in all_results["decode"]:
        print(f"{r['output_len']:10d} | {r['ttft_ms_median']:12.1f} | {r['tpot_ms_median']:12.1f} | {r['decode_throughput_tps']:18.1f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
