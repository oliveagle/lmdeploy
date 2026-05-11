#!/usr/bin/env python3
"""
STORY-004: Baseline Performance Benchmarking

Task 4.1: Run Qwen3.5-9B-AWQ baseline decode speed
- Measure single-user (chat) throughput
- Measure batch decoding throughput
"""

import os
import sys
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Configure environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

# Model path
target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"

# Test prompts - variety of lengths
short_prompts = [
    "Hello, how are you?",
    "What is Python?",
    "Explain AI in one sentence.",
    "Capital of France?",
    "Best programming language?",
]

medium_prompts = [
    "Explain the concept of machine learning and its applications in modern technology.",
    "What are the key differences between Python and JavaScript for web development?",
    "Describe the process of training a neural network from start to finish.",
]

long_prompts = [
    "Write a comprehensive guide on deep learning, covering the history, key concepts, "
    "major architectures (CNNs, RNNs, Transformers), and practical applications in computer "
    "vision, natural language processing, and speech recognition. Include examples of "
    "popular frameworks and best practices for model training and deployment.",
]


@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""
    mode: str           # "single" or "batch"
    batch_size: int     # 1 for single user
    num_requests: int
    total_tokens: int
    total_time_sec: float
    avg_latency_sec: float
    throughput_tps: float
    tokens_per_request: float
    prompt_lengths: List[int]


def check_model_exists():
    """Verify target model exists."""
    if not os.path.exists(target_model):
        print(f"ERROR: Target model not found at {target_model}")
        sys.exit(1)
    print(f"Target model: {target_model}")
    return True


def create_pipeline():
    """Create TurboMind pipeline without DFlash (baseline)."""
    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tensor_parallel=1,
        cache_max_entry_count=0.5,
        quant_policy=8,
        session_len=8192,
    )

    print("\nCreating baseline pipeline (without DFlash)...")
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        log_level='WARNING',
    )
    print("✓ Pipeline created successfully!")
    return pipe


def warmup(pipe, num_warmup=3):
    """Warmup the pipeline before benchmarking."""
    print(f"\nWarming up ({num_warmup} requests)...")
    gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)

    for i in range(num_warmup):
        prompt = short_prompts[i % len(short_prompts)]
        pipe([{"role": "user", "content": prompt}], gen_config=gen_config)
    print("✓ Warmup complete!")


def benchmark_single_user(pipe, num_requests=50, max_new_tokens=128) -> BenchmarkResult:
    """
    Benchmark single-user (chat) throughput.

    Args:
        pipe: LMDeploy pipeline
        num_requests: Number of requests to run
        max_new_tokens: Tokens to generate per request

    Returns:
        BenchmarkResult with performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Single-User (Chat) Benchmark")
    print(f"{'='*60}")
    print(f"Number of requests: {num_requests}")
    print(f"Max new tokens: {max_new_tokens}")

    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

    latencies = []
    total_tokens = 0
    prompt_lengths = []

    print(f"\n{'Idx':<6} {'Latency(s)':<12} {'OutputTokens':<14} {'TPS':<12}")
    print("-" * 50)

    for i in range(num_requests):
        prompt_idx = i % len(short_prompts)
        prompt = short_prompts[prompt_idx]

        t0 = time.time()
        resp = pipe(
            [{"role": "user", "content": prompt}],
            gen_config=gen_config,
            sequence_start=True,
            sequence_end=True,
            chat_template_kwargs={'enable_thinking': False}
        )
        t1 = time.time()

        latency = t1 - t0
        output_tokens = len(resp.token_ids) - resp.input_token_len
        total_tokens += output_tokens
        latencies.append(latency)
        prompt_lengths.append(resp.input_token_len)

        tps = output_tokens / latency if latency > 0 else 0
        print(f"{i+1:<6} {latency:<12.3f} {output_tokens:<14} {tps:<12.2f}")

        # Periodic progress
        if (i + 1) % 10 == 0:
            elapsed_so_far = sum(latencies)
            tokens_so_far = total_tokens
            avg_tps_so_far = tokens_so_far / elapsed_so_far
            print(f"  -> Progress {i+1}/{num_requests}: avg={avg_tps_so_far:.2f} tps")

    total_time = sum(latencies)
    avg_latency = total_time / num_requests
    throughput = total_tokens / total_time if total_time > 0 else 0

    result = BenchmarkResult(
        mode="single",
        batch_size=1,
        num_requests=num_requests,
        total_tokens=total_tokens,
        total_time_sec=total_time,
        avg_latency_sec=avg_latency,
        throughput_tps=throughput,
        tokens_per_request=total_tokens / num_requests,
        prompt_lengths=prompt_lengths,
    )

    print(f"\n{'='*60}")
    print(f"Single-User Benchmark Results")
    print(f"{'='*60}")
    print(f"Total requests:     {result.num_requests}")
    print(f"Total tokens:       {result.total_tokens}")
    print(f"Total time:         {result.total_time_sec:.2f}s")
    print(f"Avg latency:        {result.avg_latency_sec:.3f}s/request")
    print(f"Throughput:         {result.throughput_tps:.2f} tokens/s")
    print(f"Avg tokens/req:     {result.tokens_per_request:.1f}")
    print(f"{'='*60}")

    return result


def benchmark_batch(pipe, batch_sizes=[4, 8, 16], requests_per_batch=20, max_new_tokens=128) -> Dict[int, BenchmarkResult]:
    """
    Benchmark batch decoding throughput.

    Args:
        pipe: LMDeploy pipeline
        batch_sizes: List of batch sizes to test
        requests_per_batch: Number of requests per batch size
        max_new_tokens: Tokens to generate per request

    Returns:
        Dict mapping batch size to BenchmarkResult
    """
    all_results = {}

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch Decoding Benchmark (batch_size={batch_size})")
        print(f"{'='*60}")
        print(f"Number of batches: {requests_per_batch}")
        print(f"Max new tokens: {max_new_tokens}")

        gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

        batch_latencies = []
        total_tokens = 0
        prompt_lengths = []

        print(f"\n{'Batch':<8} {'Latency(s)':<12} {'OutputTokens':<14} {'TPS':<12}")
        print("-" * 54)

        for batch_idx in range(requests_per_batch):
            # Build batch prompts
            batch_prompts = []
            for i in range(batch_size):
                prompt_idx = (batch_idx * batch_size + i) % len(medium_prompts)
                batch_prompts.append([{"role": "user", "content": medium_prompts[prompt_idx]}])

            t0 = time.time()
            responses = pipe(
                batch_prompts,
                gen_config=gen_config,
                sequence_start=True,
                sequence_end=True,
                chat_template_kwargs={'enable_thinking': False}
            )
            t1 = time.time()

            latency = t1 - t0
            batch_output_tokens = 0
            for resp in responses:
                output_tokens = len(resp.token_ids) - resp.input_token_len
                batch_output_tokens += output_tokens
                prompt_lengths.append(resp.input_token_len)

            total_tokens += batch_output_tokens
            batch_latencies.append(latency)

            tps = batch_output_tokens / latency if latency > 0 else 0
            print(f"{batch_idx+1:<8} {latency:<12.3f} {batch_output_tokens:<14} {tps:<12.2f}")

        total_time = sum(batch_latencies)
        total_requests = requests_per_batch * batch_size
        avg_latency = total_time / requests_per_batch  # avg per batch
        throughput = total_tokens / total_time if total_time > 0 else 0

        result = BenchmarkResult(
            mode="batch",
            batch_size=batch_size,
            num_requests=total_requests,
            total_tokens=total_tokens,
            total_time_sec=total_time,
            avg_latency_sec=avg_latency,
            throughput_tps=throughput,
            tokens_per_request=total_tokens / total_requests,
            prompt_lengths=prompt_lengths,
        )

        all_results[batch_size] = result

        print(f"\n{'='*60}")
        print(f"Batch Benchmark Results (batch_size={batch_size})")
        print(f"{'='*60}")
        print(f"Total requests:     {result.num_requests}")
        print(f"Total tokens:       {result.total_tokens}")
        print(f"Total time:         {result.total_time_sec:.2f}s")
        print(f"Avg batch latency:  {result.avg_latency_sec:.3f}s/batch")
        print(f"Throughput:         {result.throughput_tps:.2f} tokens/s")
        print(f"Avg tokens/req:     {result.tokens_per_request:.1f}")
        print(f"{'='*60}")

    return all_results


def save_results(single_result: BenchmarkResult, batch_results: Dict[int, BenchmarkResult], output_file="baseline_benchmark_results.json"):
    """Save benchmark results to JSON file."""
    results_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": target_model,
        "single_user": asdict(single_result),
        "batch": {str(bs): asdict(r) for bs, r in batch_results.items()},
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


def print_summary(single_result: BenchmarkResult, batch_results: Dict[int, BenchmarkResult]):
    """Print a summary of all benchmarks."""
    print(f"\n{'='*60}")
    print(f"BASELINE BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {target_model}")
    print(f"\nSingle-user: {single_result.throughput_tps:.2f} tokens/s")
    for bs, r in batch_results.items():
        print(f"Batch {bs:2d}:    {r.throughput_tps:.2f} tokens/s")
    print(f"{'='*60}")


def main():
    print("="*60)
    print("STORY-004: Baseline Performance Benchmarking")
    print("="*60)

    # Check model
    check_model_exists()

    # Create pipeline
    pipe = create_pipeline()

    # Warmup
    warmup(pipe, num_warmup=3)

    # Run single-user benchmark
    single_result = benchmark_single_user(pipe, num_requests=30, max_new_tokens=128)

    # Run batch benchmarks
    batch_results = benchmark_batch(pipe, batch_sizes=[4, 8], requests_per_batch=10, max_new_tokens=128)

    # Save and print summary
    save_results(single_result, batch_results)
    print_summary(single_result, batch_results)

    # Cleanup
    del pipe

    print("\n✓ Baseline benchmark complete!")


if __name__ == "__main__":
    main()
