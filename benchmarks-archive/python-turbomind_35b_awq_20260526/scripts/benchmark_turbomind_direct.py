#!/usr/bin/env python3
"""
LMDeploy TurboMind Direct Benchmark Script

直接调用 TurboMind 引擎进行性能测试，与 Rust benchmark 口径一致。
不使用 HTTP API，避免网络延迟开销。

测量方式：
- TTFT: 从请求开始到第一个 token 返回的时间
- Prefill throughput: input_tokens / ttft
- Decode throughput: output_tokens / (total_time - ttft)
"""
import asyncio
import json
import time
from datetime import datetime

from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig

# 配置
MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

# 测试场景
SCENARIOS = [
    {"name": "short_context", "input_len": 512, "output_len": 512},
    {"name": "medium_context", "input_len": 1024, "output_len": 512},
    {"name": "long_context", "input_len": 4096, "output_len": 512},
    {"name": "long_context", "input_len": 8192, "output_len": 512},
]

NUM_REQUESTS = 5  # 每个场景请求数
WARMUP_REQUESTS = 1


def generate_prompt(tokenizer, target_len):
    """生成指定 token 长度的 prompt"""
    base_tokens = tokenizer.encode("Hello, how are you today?")
    repeat_times = (target_len // len(base_tokens)) + 1
    input_ids = (base_tokens * repeat_times)[:target_len]
    return tokenizer.decode(input_ids), input_ids


async def run_single_benchmark(engine, input_ids, output_len, session_id):
    """运行单个 benchmark 请求，返回详细指标"""
    params = GenerationConfig(
        max_new_tokens=output_len,
        ignore_eos=True,
    )

    start = time.perf_counter()
    ttft = None
    token_count = 0
    token_times = []

    try:
        async for outputs in engine.async_stream_infer(
            session_id,
            input_ids=input_ids,
            gen_config=params,
            sequence_start=True,
            sequence_end=True,
            stream_output=True,
        ):
            now = time.perf_counter()

            # 记录第一个 token 时间 (TTFT)
            if ttft is None and outputs.token_ids:
                ttft = now - start

            # 记录每个 token 的时间
            for _ in outputs.token_ids:
                token_count += 1
                token_times.append(now - start)

        total_time = time.perf_counter() - start

        return {
            "ttft": ttft,
            "total_time": total_time,
            "decode_time": total_time - ttft if ttft else 0,
            "token_count": token_count,
            "token_times": token_times,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None


async def benchmark_scenario(engine, tokenizer, scenario, iteration_offset=0):
    """对一个场景进行基准测试"""
    input_len = scenario["input_len"]
    output_len = scenario["output_len"]

    # 生成指定长度的 prompt
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
        await run_single_benchmark(engine, input_ids, 32, i)
    await asyncio.sleep(1)

    # Actual benchmark - 串行执行
    print(f"Benchmark: {NUM_REQUESTS} requests...")
    for i in range(NUM_REQUESTS):
        session_id = iteration_offset + i
        result = await run_single_benchmark(engine, input_ids, output_len, session_id)
        if result:
            results.append(result)
            print(f"  Run {i+1}: TTFT={result['ttft']*1000:.1f}ms, Decode={result['decode_time']*1000:.1f}ms, Tokens={result['token_count']}")
        # 使用 sequence_start=True, sequence_end=True 是一次性请求，不需要 async_end

    # 统计结果
    if not results:
        print("No valid results!")
        return None

    # 计算平均指标
    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    avg_total_time = sum(r["total_time"] for r in results) / len(results)
    avg_decode_time = sum(r["decode_time"] for r in results) / len(results)

    # 与 Rust benchmark 一致的计算方式
    # Prefill throughput: input_tokens / ttft
    prefill_tps = actual_input_len / avg_ttft if avg_ttft > 0 else 0

    # Decode throughput: output_tokens / decode_time
    avg_output_tokens = sum(r["token_count"] for r in results) / len(results)
    decode_tps = avg_output_tokens / avg_decode_time if avg_decode_time > 0 else 0

    stats = {
        "scenario": scenario["name"],
        "input_len": actual_input_len,
        "output_len": output_len,
        "completed": len(results),
        "ttft_ms_avg": avg_ttft * 1000,
        "prefill_time_ms_avg": avg_ttft * 1000,
        "decode_time_ms_avg": avg_decode_time * 1000,
        "total_time_ms_avg": avg_total_time * 1000,
        "prefill_throughput_tps": prefill_tps,
        "decode_throughput_tps": decode_tps,
        "avg_output_tokens": avg_output_tokens,
    }

    print(f"\nResults:")
    print(f"  Completed: {stats['completed']}/{NUM_REQUESTS}")
    print(f"  TTFT avg: {stats['ttft_ms_avg']:.2f} ms")
    print(f"  Prefill time avg: {stats['prefill_time_ms_avg']:.2f} ms")
    print(f"  Decode time avg: {stats['decode_time_ms_avg']:.2f} ms")
    print(f"  Prefill throughput: {stats['prefill_throughput_tps']:.2f} tok/s")
    print(f"  Decode throughput: {stats['decode_throughput_tps']:.2f} tok/s")

    return stats


async def main():
    """主函数"""
    print(f"{'='*60}")
    print(f"LMDeploy TurboMind Direct Benchmark")
    print(f"Model: {MODEL_PATH}")
    print(f"Method: Direct engine call (no HTTP)")
    print(f"{'='*60}")

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    # 初始化 TurboMind 引擎
    print("Initializing TurboMind engine...")
    tm_model = TurboMind.from_pretrained(MODEL_PATH)
    model_inst = tm_model.create_instance()

    # 预热
    print("Warming up engine...")
    try:
        async for _ in model_inst.async_stream_infer(
            0,
            input_ids=tokenizer.encode("Hello"),
            gen_config=GenerationConfig(max_new_tokens=10),
            sequence_start=True,
            sequence_end=True,
            stream_output=False,
        ):
            pass
        print("Warmup complete\n")
    except Exception as e:
        print(f"Warmup warning: {e}")

    # 运行所有场景
    all_results = []
    session_id = 100  # 从较高 ID 开始避免冲突

    for scenario in SCENARIOS:
        result = await benchmark_scenario(tm_model, tokenizer, scenario, session_id)
        if result:
            all_results.append(result)
        session_id += NUM_REQUESTS + 10
        await asyncio.sleep(2)  # 场景间隔

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_turbomind_direct_{timestamp}.json"

    output = {
        "engine": "LMDeploy TurboMind (direct)",
        "model": MODEL_PATH,
        "backend": "turbomind",
        "method": "direct_engine_call",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lmdeploy_version": "0.13.0",
        "results": {r["scenario"]: r for r in all_results},
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # 打印摘要
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':30s} | {'TTFT (ms)':>10s} | {'Prefill (tok/s)':>18s} | {'Decode (tok/s)':>18s}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['scenario']:30s} | {r['ttft_ms_avg']:10.2f} | {r['prefill_throughput_tps']:18.2f} | {r['decode_throughput_tps']:18.2f}")

    # 与存档数据对比
    print(f"\n{'='*80}")
    print("对比存档数据 (profile_throughput_35b_random_1024_512.csv)")
    print(f"{'='*80}")
    print("存档: TTFT=139ms, Prefill=7356 tok/s, Decode=42.2 tok/s")
    for r in all_results:
        if r['input_len'] == 1024:
            print(f"当前: TTFT={r['ttft_ms_avg']:.1f}ms, Prefill={r['prefill_throughput_tps']:.1f} tok/s, Decode={r['decode_throughput_tps']:.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
