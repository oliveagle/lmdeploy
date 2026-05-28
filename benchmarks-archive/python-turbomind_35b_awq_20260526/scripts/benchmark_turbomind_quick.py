#!/usr/bin/env python3
"""
LMDeploy TurboMind Benchmark Script
测试 Python + TurboMind 后端的性能表现
"""
import asyncio
import json
import time
import subprocess
import requests
import aiohttp
from datetime import datetime

# 配置
MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
PORT = 23333
BACKEND = "turbomind"

# 测试场景
SCENARIOS = [
    {"name": "short_context", "input_len": 1024, "output_len": 512},
    {"name": "medium_context", "input_len": 2048, "output_len": 512},
    {"name": "long_context", "input_len": 4096, "output_len": 512},
    {"name": "long_context", "input_len": 8192, "output_len": 512},
    {"name": "long_context_large_output", "input_len": 8192, "output_len": 2048},
]

NUM_REQUESTS = 10  # 每个场景请求数
WARMUP_REQUESTS = 2


def start_server():
    """启动 TurboMind 服务器"""
    cmd = [
        "lmdeploy", "serve", "api_server",
        MODEL_PATH,
        "--backend", BACKEND,
        "--server-name", "0.0.0.0",
        "--server-port", str(PORT),
    ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def get_model_name():
    """获取模型名称"""
    url = f"http://localhost:{PORT}/v1/models"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception as e:
        print(f"Failed to get model name: {e}")
    return None


def wait_server_ready(max_wait=120):
    """等待服务器就绪"""
    url = f"http://localhost:{PORT}/v1/models"
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    print(f"Server ready. Model: {models[0]['id']}")
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_server(proc):
    """停止服务器"""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


async def run_single_request(session, prompt, output_len, model_name):
    """执行单个请求并返回结果"""
    url = f"http://localhost:{PORT}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
    }

    headers = {"Authorization": "Bearer DUMMY"}

    ttft = None
    itls = []
    total_time = None
    generated_text = ""
    last_ts = None

    try:
        start = time.perf_counter()

        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"  Error: status={resp.status}, body={text[:200]}")
                return None

            async for chunk_bytes in resp.content:
                chunk = chunk_bytes.decode().strip()
                if not chunk or chunk == "data: " or chunk == "data:":
                    continue

                if chunk.startswith("data: "):
                    chunk = chunk[6:]

                if chunk == "[DONE]":
                    total_time = time.perf_counter() - start
                    break

                try:
                    data = json.loads(chunk)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                        else:
                            if last_ts is not None:
                                itls.append(now - last_ts)
                        last_ts = now
                        generated_text += content

                except json.JSONDecodeError:
                    continue

            if total_time is None:
                total_time = time.perf_counter() - start

    except Exception as e:
        print(f"Request error: {e}")
        return None

    return {
        "ttft": ttft,
        "itls": itls,
        "total_time": total_time,
        "generated_len": len(generated_text),
    }


async def benchmark_scenario(session, scenario, num_requests, model_name, tokenizer):
    """对一个场景进行基准测试"""

    # 生成指定长度的 prompt
    if tokenizer:
        base_tokens = tokenizer.encode("Hello, how are you today?")
        repeat_times = (scenario["input_len"] // len(base_tokens)) + 1
        input_ids = (base_tokens * repeat_times)[:scenario["input_len"]]
        prompt = tokenizer.decode(input_ids)
    else:
        prompt = "Hello. " * (scenario["input_len"] // 10)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['name']}")
    print(f"Input length: {scenario['input_len']}, Output length: {scenario['output_len']}")
    print(f"{'='*60}")

    results = []

    # Warmup
    print(f"Warmup: {WARMUP_REQUESTS} requests...")
    for _ in range(WARMUP_REQUESTS):
        await run_single_request(session, prompt[:500], 32, model_name)
    await asyncio.sleep(2)

    # Actual benchmark
    print(f"Benchmark: {num_requests} requests...")
    tasks = []
    for _ in range(num_requests):
        task = run_single_request(session, prompt, scenario["output_len"], model_name)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # 统计结果
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("No valid results!")
        return None

    ttfts = [r["ttft"] for r in valid_results if r["ttft"] is not None]
    itls_flat = [itl for r in valid_results for itl in r["itls"]]
    total_times = [r["total_time"] for r in valid_results if r["total_time"] is not None]

    # 分离 prefill 和 decode 阶段的指标
    # Prefill 阶段：TTFT 时间内的 token 处理 = input_len 个 prompt tokens
    # Decode 阶段：(total_time - TTFT) 时间内生成 output_len 个 token
    prefill_times = ttfts  # 每个请求的 prefill 时间 = TTFT
    decode_times = []
    for r in valid_results:
        if r["ttft"] is not None and r["total_time"] is not None:
            decode_time = r["total_time"] - r["ttft"]
            decode_times.append(decode_time)

    # Prefill 吞吐量 = 总输入 tokens / 总 prefill 时间
    total_prefill_time = sum(prefill_times) if prefill_times else 0
    prefill_tps = len(valid_results) * scenario["input_len"] / total_prefill_time if total_prefill_time > 0 else 0

    # Decode 吞吐量 = 总输出 tokens / 总 decode 时间
    total_decode_time = sum(decode_times) if decode_times else 0
    decode_tps = len(valid_results) * scenario["output_len"] / total_decode_time if total_decode_time > 0 else 0

    stats = {
        "scenario": scenario["name"],
        "input_len": scenario["input_len"],
        "output_len": scenario["output_len"],
        "completed": len(valid_results),
        "ttft_ms_avg": sum(ttfts) / len(ttfts) * 1000 if ttfts else 0,
        "ttft_ms_p99": sorted(ttfts)[int(len(ttfts) * 0.99)] if ttfts else 0,
        "itl_ms_avg": sum(itls_flat) / len(itls_flat) * 1000 if itls_flat else 0,
        "total_time_ms_avg": sum(total_times) / len(total_times) * 1000 if total_times else 0,
        "prefill_time_ms_avg": total_prefill_time / len(prefill_times) * 1000 if prefill_times else 0,
        "decode_time_ms_avg": sum(decode_times) / len(decode_times) * 1000 if decode_times else 0,
        "prefill_throughput_tps": prefill_tps,
        "decode_throughput_tps": decode_tps,
        "throughput_tps": len(valid_results) * scenario["output_len"] / (sum(total_times) if total_times else 1),
    }

    print(f"\nResults:")
    print(f"  Completed: {stats['completed']}/{num_requests}")
    print(f"  TTFT avg: {stats['ttft_ms_avg']:.2f} ms (P99: {stats['ttft_ms_p99']:.2f} ms)")
    print(f"  ITL avg: {stats['itl_ms_avg']:.2f} ms")
    print(f"  Prefill time avg: {stats['prefill_time_ms_avg']:.2f} ms")
    print(f"  Decode time avg: {stats['decode_time_ms_avg']:.2f} ms")
    print(f"  Prefill throughput: {stats['prefill_throughput_tps']:.2f} tok/s")
    print(f"  Decode throughput: {stats['decode_throughput_tps']:.2f} tok/s")
    print(f"  Total time avg: {stats['total_time_ms_avg']:.2f} ms")

    return stats


async def main():
    """主函数"""
    print(f"{'='*60}")
    print(f"LMDeploy TurboMind Benchmark")
    print(f"Model: {MODEL_PATH}")
    print(f"Backend: {BACKEND}")
    print(f"{'='*60}")

    # 启动服务器
    proc = start_server()

    try:
        if not wait_server_ready():
            print("Server failed to start!")
            return

        model_name = get_model_name()
        if not model_name:
            print("Failed to get model name!")
            return

        print(f"Using model: {model_name}")

        # 加载 tokenizer
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")

        # 创建 session
        async with aiohttp.ClientSession() as session:
            # 运行所有场景
            all_results = []
            for scenario in SCENARIOS:
                result = await benchmark_scenario(session, scenario, NUM_REQUESTS, model_name, tokenizer)
                if result:
                    all_results.append(result)
                await asyncio.sleep(5)  # 场景间隔

    finally:
        stop_server(proc)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_turbomind_{timestamp}.json"

    output = {
        "engine": "LMDeploy TurboMind",
        "model": MODEL_PATH,
        "backend": BACKEND,
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


if __name__ == "__main__":
    asyncio.run(main())