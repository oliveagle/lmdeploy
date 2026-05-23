#!/usr/bin/env python3
"""
LMDeploy TurboMind 真实性能测试
区分 prefill 和 decode 阶段，测量实际吞吐量
"""

import time
import requests
import json
import statistics
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    context_len: int
    prompt_tokens: int
    output_tokens: int
    prefill_time: float  # ms
    decode_time: float  # ms
    first_token_latency: float  # ms
    prefill_throughput: float  # tokens/s
    decode_throughput: float  # tokens/s

class TurboMindBenchmark:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.model = "/mnt/data/models/modelscope_models/Qwen3.6-35B-A3B-AWQ"

    def create_chat_payload(self, prompt: str, max_tokens: int = 128) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False
        }

    def run_stream_inference(self, prompt: str, max_tokens: int = 128) -> Dict:
        """使用流式 API 获取 first_token_latency"""
        payload = self.create_chat_payload(prompt, max_tokens)
        payload["stream"] = True

        start_time = time.time()
        first_token_time = None
        token_count = 0

        resp = requests.post(f"{self.base_url}/v1/chat/completions",
                              json=payload, stream=True, timeout=300)

        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                token_count += len(delta['content'])
                    except json.JSONDecodeError:
                        pass

        end_time = time.time()

        total_time = (end_time - start_time) * 1000  # ms

        if first_token_time:
            first_token_latency = (first_token_time - start_time) * 1000  # ms
            decode_time = total_time - first_token_latency
        else:
            first_token_latency = 0
            decode_time = total_time

        return {
            "prompt_tokens": token_count,  # 简化，实际需要 tokenizer
            "output_tokens": token_count,
            "first_token_latency": first_token_latency,
            "decode_time": decode_time,
            "total_time": total_time
        }

    def benchmark_context_size(self, context_len: int, num_runs: int = 3) -> BenchmarkResult:
        """测试指定 context 长度的性能"""
        print(f"\n测试 Context {context_len}...")

        # 构造指定长度的 prompt
        base_prompt = "请用一句话回答：什么是人工智能？"
        padding = "A" * (context_len - len(base_prompt))
        prompt = base_prompt + padding

        results = []

        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end=" ", flush=True)

            if i == 0:
                time.sleep(1)  # 额外等待确保服务就绪

            result = self.run_stream_inference(prompt, max_tokens=128)

            prompt_toks = result["prompt_tokens"]
            output_toks = result["output_tokens"]
            first_token_latency = result["first_token_latency"]
            total_time = result["total_time"]

            # 估算: first token 之前是 prefill
            prefill_time = first_token_latency if first_token_latency > 0 else total_time * 0.1
            decode_time = total_time - prefill_time

            prefill_throughput = (prompt_toks / prefill_time * 1000) if prefill_time > 0 else 0
            decode_throughput = (output_toks / decode_time * 1000) if decode_time > 0 else 0

            print(f"Prompt: {prompt_toks}, Output: {output_toks}, "
                  f"FirstToken: {first_token_latency:.0f}ms, "
                  f"Prefill: {prefill_throughput:.0f} t/s, Decode: {decode_throughput:.0f} t/s")

            results.append({
                "prompt_tokens": prompt_toks,
                "output_tokens": output_toks,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "prefill_throughput": prefill_throughput,
                "decode_throughput": decode_throughput
            })

        # 计算平均值
        avg = {
            "context_len": context_len,
            "prompt_tokens": statistics.mean([r["prompt_tokens"] for r in results]),
            "output_tokens": statistics.mean([r["output_tokens"] for r in results]),
            "prefill_time": statistics.mean([r["prefill_time"] for r in results]),
            "decode_time": statistics.mean([r["decode_time"] for r in results]),
            "first_token_latency": statistics.mean([r.get("first_token_latency", 0) for r in results]),
            "prefill_throughput": statistics.mean([r["prefill_throughput"] for r in results]),
            "decode_throughput": statistics.mean([r["decode_throughput"] for r in results]),
        }

        return BenchmarkResult(**avg)

    def run_full_benchmark(self, context_sizes: List[int] = None, num_runs: int = 3):
        """运行完整基准测试"""
        if context_sizes is None:
            context_sizes = [128, 512, 1024, 2048, 4096]

        print("=" * 80)
        print("LMDeploy TurboMind 性能基准测试")
        print("=" * 80)

        results = []
        for ctx_size in context_sizes:
            result = self.benchmark_context_size(ctx_size, num_runs)
            results.append(result)

        # 输出汇总
        print("\n" + "=" * 80)
        print("性能汇总 (平均值)")
        print("=" * 80)
        print(f"{'Context':<10} {'Prefill':<12} {'Decode':<12} {'FirstToken':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r.context_len:<10} {r.prefill_throughput:<12.0f} {r.decode_throughput:<12.0f} {r.first_token_latency:<12.0f}")

        return results

if __name__ == "__main__":
    import sys

    benchmark = TurboMindBenchmark()

    if len(sys.argv) > 1:
        ctx_size = int(sys.argv[1])
        benchmark.benchmark_context_size(ctx_size, num_runs=5)
    else:
        benchmark.run_full_benchmark()
