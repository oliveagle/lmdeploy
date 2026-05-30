#!/usr/bin/env python3
"""
LMDeploy TurboMind Direct Benchmark (Sync)

使用同步 API 避免 async_stream_infer 的段错误问题
"""
import time
import json
from datetime import datetime

from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig

MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

SCENARIOS = [
    {"name": "short_context", "input_len": 512, "output_len": 512},
    {"name": "medium_context", "input_len": 1024, "output_len": 512},
    {"name": "long_context", "input_len": 4096, "output_len": 512},
    {"name": "long_context", "input_len": 8192, "output_len": 512},
]

NUM_REQUESTS = 5
WARMUP_REQUESTS = 1


def generate_prompt(tokenizer, target_len):
    """生成指定 token 长度的 prompt"""
    base_tokens = tokenizer.encode("Hello, how are you today?")
    repeat_times = (target_len // len(base_tokens)) + 1
    input_ids = (base_tokens * repeat_times)[:target_len]
    return tokenizer.decode(input_ids), input_ids


def run_single_benchmark_sync(engine, input_ids, output_len, session_id):
    """运行单个 benchmark - 同步方式"""
    params = GenerationConfig(
        max_new_tokens=output_len,
        ignore_eos=True,
    )

    start = time.perf_counter()

    try:
        # 使用同步接口
        outputs = engine.infer(
            session_id,
            input_ids=input_ids,
            gen_config=params,
            sequence_start=True,
            sequence_end=True,
        )

        total_time = time.perf_counter() - start

        # 同步接口没有 TTFT 信息，只能用总时间估算
        # 假设 TTFT 约占总时间的 10%（这是一个近似值）
        estimated_ttft = total_time * 0.1
        estimated_decode_time = total_time - estimated_ttft

        return {
            "total_time": total_time,
            "estimated_ttft": estimated_ttft,
            "estimated_decode_time": estimated_decode_time,
            "token_count": len(outputs.token_ids),
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None


def benchmark_scenario_sync(engine, tokenizer, scenario):
    """对一个场景进行基准测试 - 同步方式"""
    input_len = scenario["input_len"]
    output_len = scenario["output_len"]

    prompt, input_ids = generate_prompt(tokenizer, input_len)
    actual_input_len = len(input_ids)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['name']}")
    print(f"Input length: {actual_input_len}, Output length: {output_len}")
    print(f"{'='*60}")

    results = []

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} requests...")
    for i in range(WARMUP_REQUESTS):
        run_single_benchmark_sync(engine, input_ids, 32, i)

    # Actual benchmark
    print(f"Benchmark: {NUM_REQUESTS} requests...")
    for i in range(NUM_REQUESTS):
        result = run_single_benchmark_sync(engine, input_ids, output_len, i)
        if result:
            results.append(result)
            print(f"  Run {i+1}: Total={result['total_time']*1000:.1f}ms, Tokens={result['token_count']}")

    # 统计结果
    if not results:
        print("No valid results!")
        return None

    avg_total_time = sum(r["total_time"] for r in results) / len(results)
    avg_estimated_ttft = sum(r["estimated_ttft"] for r in results) / len(results)
    avg_estimated_decode_time = sum(r["estimated_decode_time"] for r in results) / len(results)
    avg_output_tokens = sum(r["token_count"] for r in results) / len(results)

    prefill_tps = actual_input_len / avg_estimated_ttft if avg_estimated_ttft > 0 else 0
    decode_tps = avg_output_tokens / avg_estimated_decode_time if avg_estimated_decode_time > 0 else 0

    stats = {
        "scenario": scenario["name"],
        "input_len": actual_input_len,
        "output_len": output_len,
        "completed": len(results),
        "ttft_ms_avg": avg_estimated_ttft * 1000,
        "total_time_ms_avg": avg_total_time * 1000,
        "decode_time_ms_avg": avg_estimated_decode_time * 1000,
        "prefill_throughput_tps": prefill_tps,
        "decode_throughput_tps": decode_tps,
        "avg_output_tokens": avg_output_tokens,
    }

    print(f"\nResults:")
    print(f"  Completed: {stats['completed']}/{NUM_REQUESTS}")
    print(f"  Total time avg: {stats['total_time_ms_avg']:.2f} ms")
    print(f"  Est. TTFT: {stats['ttft_ms_avg']:.2f} ms")
    print(f"  Prefill throughput: {stats['prefill_throughput_tps']:.2f} tok/s")
    print(f"  Decode throughput: {stats['decode_throughput_tps']:.2f} tok/s")

    return stats


def main():
    print(f"{'='*60}")
    print(f"LMDeploy TurboMind Direct Benchmark (Sync)")
    print(f"Model: {MODEL_PATH}")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    print("Initializing TurboMind engine...")
    tm_model = TurboMind.from_pretrained(MODEL_PATH)
    model_inst = tm_model.create_instance()

    print("Warming up...")
    try:
        model_inst.infer(
            0,
            input_ids=tokenizer.encode("Hello"),
            gen_config=GenerationConfig(max_new_tokens=10),
            sequence_start=True,
            sequence_end=True,
        )
        print("Warmup complete\n")
    except Exception as e:
        print(f"Warmup warning: {e}")

    all_results = []

    for scenario in SCENARIOS:
        result = benchmark_scenario_sync(model_inst, tokenizer, scenario)
        if result:
            all_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_turbomind_sync_{timestamp}.json"

    output = {
        "engine": "LMDeploy TurboMind (sync)",
        "model": MODEL_PATH,
        "backend": "turbomind",
        "method": "sync_infer",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lmdeploy_version": "0.13.0",
        "results": {r["scenario"]: r for r in all_results},
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':30s} | {'Total (ms)':>10s} | {'Prefill (tok/s)':>18s} | {'Decode (tok/s)':>18s}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['scenario']:30s} | {r['total_time_ms_avg']:10.2f} | {r['prefill_throughput_tps']:18.2f} | {r['decode_throughput_tps']:18.2f}")

    print(f"\n{'='*80}")
    print("对比存档数据 (profile_throughput_35b_random_1024_512.csv)")
    print(f"{'='*80}")
    print("存档: TTFT=139ms, Prefill=7356 tok/s, Decode=42.2 tok/s")
    for r in all_results:
        if r['input_len'] == 1024:
            print(f"当前: Total={r['total_time_ms_avg']:.1f}ms, Est. TTFT={r['ttft_ms_avg']:.1f}ms, Prefill={r['prefill_throughput_tps']:.1f} tok/s, Decode={r['decode_throughput_tps']:.1f} tok/s")


if __name__ == "__main__":
    main()
