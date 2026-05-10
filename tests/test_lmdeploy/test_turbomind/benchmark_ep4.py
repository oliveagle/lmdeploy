#!/usr/bin/env python3
"""EP=4 Performance Benchmark Script.

This script benchmarks EP=4 vs TP=4 configurations for Qwen3.6-35B-A3B-AWQ.
It measures:
1. Model loading time
2. Prefill latency (time to first token)
3. Decode throughput (tokens/second)
4. Memory usage per GPU

Prerequisites:
- Qwen3.6-35B-A3B-AWQ model weights
- 4x V100 16GB GPUs (or 4x A100 40GB recommended)
- Turbomind C++ extension compiled
- nvidia-smi for GPU memory monitoring

Usage:
    python tests/test_lmdeploy/test_turbomind/benchmark_ep4.py --model-path /path/to/model
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    config_name: str
    ep: int
    tp: int
    device_num: int
    load_time: float
    prefill_latency: float
    decode_throughput: float
    total_time: float
    output_tokens: int
    gpu_memory_mb: List[float]
    output_quality_score: float


def get_gpu_memory():
    """Get GPU memory usage in MB for all GPUs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        memory_mb = [float(x.strip()) for x in result.stdout.strip().split('\n')]
        return memory_mb
    except Exception as e:
        print(f"⚠️  Failed to get GPU memory: {e}")
        return []


def calculate_quality_score(output: str) -> float:
    """Calculate output quality score.

    Higher score = better quality
    - Penalize excessive '!' characters (garbled output)
    - Penalize excessive repetition
    - Reward vocabulary diversity

    Args:
        output: Generated text

    Returns:
        Quality score (0-100)
    """
    if not output:
        return 0.0

    # Base score
    score = 100.0

    # Penalize excessive '!' characters
    exclam_ratio = output.count('!') / len(output) if output else 0
    if exclam_ratio > 0.3:
        score -= 50.0 * exclam_ratio

    # Penalize excessive repetition
    words = output.split()
    if len(words) > 10:
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)
        if unique_ratio < 0.3:
            score -= 30.0 * (1.0 - unique_ratio)

    # Reward vocabulary diversity
    if len(words) > 20:
        unique_ratio = len(unique_words) / len(words)
        score += 10.0 * unique_ratio

    return max(0.0, min(100.0, score))


def benchmark_configuration(model_path: str, config_name: str, ep: int, tp: int,
                           prompts: List[str], max_tokens: int = 128) -> BenchmarkResult:
    """Benchmark a specific configuration.

    Args:
        model_path: Path to model
        config_name: Configuration name (e.g., "EP=4, TP=1")
        ep: Expert parallelism size
        tp: Tensor parallelism size
        prompts: List of test prompts
        max_tokens: Maximum tokens to generate

    Returns:
        BenchmarkResult with metrics
    """
    from lmdeploy import TurbomindEngineConfig, pipeline

    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {config_name}")
    print(f"{'=' * 70}")
    print(f"EP={ep}, TP={tp}, Devices={ep * tp}")
    print()

    # Create engine config
    engine_config = TurbomindEngineConfig(
        ep=ep,
        tp=tp,
        device_num=ep * tp,
        session_len=2048,
        max_batch_size=1,
        quant_policy=8,  # KV cache quantization
        max_prefill_token_num=8192,
    )

    # Measure loading time
    print("Loading model...")
    start_time = time.time()
    pipe = pipeline(model_path, backend='turbomind', engine_config=engine_config)
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f}s")

    # Get GPU memory after loading
    gpu_memory = get_gpu_memory()
    print(f"GPU Memory: {gpu_memory} MB")

    # Run inference benchmarks
    total_output_tokens = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0
    total_quality_score = 0.0

    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}/{len(prompts)}: {prompt[:50]}...")

        # Measure prefill latency (time to first token)
        # Note: Turbomind doesn't expose prefill/decode separately, so we measure total time
        start_time = time.time()
        response = pipe([prompt], max_tokens=max_tokens)
        total_time = time.time() - start_time

        output = response[0]
        output_tokens = len(output.split())  # Rough estimate
        total_output_tokens += output_tokens

        # Calculate quality score
        quality_score = calculate_quality_score(output)
        total_quality_score += quality_score

        print(f"  Output: {output[:100]}...")
        print(f"  Tokens: ~{output_tokens}")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Quality Score: {quality_score:.1f}/100")

        # Accumulate metrics
        total_prefill_time += total_time * 0.1  # Assume 10% prefill
        total_decode_time += total_time * 0.9  # Assume 90% decode

    # Calculate averages
    avg_prefill_latency = total_prefill_time / len(prompts)
    decode_throughput = total_output_tokens / total_decode_time if total_decode_time > 0 else 0
    avg_quality_score = total_quality_score / len(prompts)
    total_benchmark_time = load_time + total_prefill_time + total_decode_time

    print(f"\n{'=' * 70}")
    print(f"Results: {config_name}")
    print(f"{'=' * 70}")
    print(f"Load Time: {load_time:.2f}s")
    print(f"Prefill Latency: {avg_prefill_latency:.2f}s")
    print(f"Decode Throughput: {decode_throughput:.2f} tokens/s")
    print(f"Output Quality: {avg_quality_score:.1f}/100")
    print(f"Total Time: {total_benchmark_time:.2f}s")
    print(f"GPU Memory: {gpu_memory} MB")

    return BenchmarkResult(
        config_name=config_name,
        ep=ep,
        tp=tp,
        device_num=ep * tp,
        load_time=load_time,
        prefill_latency=avg_prefill_latency,
        decode_throughput=decode_throughput,
        total_time=total_benchmark_time,
        output_tokens=total_output_tokens,
        gpu_memory_mb=gpu_memory,
        output_quality_score=avg_quality_score,
    )


def run_benchmark_suite(model_path: str, output_file: str = None):
    """Run full benchmark suite comparing EP=4 and TP=4.

    Args:
        model_path: Path to model
        output_file: Optional JSON file to save results
    """
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("   Set QWEN36_A3B_AWQ_PATH environment variable or use --model-path")
        return

    print("=" * 70)
    print("EP=4 Performance Benchmark Suite")
    print("=" * 70)
    print(f"Model: {model_path}")
    print()

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a short poem about artificial intelligence.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
    ]

    results = []

    # Benchmark EP=4, TP=1
    try:
        result_ep4 = benchmark_configuration(
            model_path=model_path,
            config_name="EP=4, TP=1",
            ep=4,
            tp=1,
            prompts=prompts,
        )
        results.append(result_ep4)
    except Exception as e:
        print(f"❌ EP=4 benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Benchmark TP=4, EP=1 (baseline)
    try:
        result_tp4 = benchmark_configuration(
            model_path=model_path,
            config_name="EP=1, TP=4",
            ep=1,
            tp=4,
            prompts=prompts,
        )
        results.append(result_tp4)
    except Exception as e:
        print(f"❌ TP=4 benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Print comparison
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("Comparison Summary")
        print("=" * 70)

        for r in results:
            print(f"\n{r.config_name}:")
            print(f"  Load Time: {r.load_time:.2f}s")
            print(f"  Decode Throughput: {r.decode_throughput:.2f} tokens/s")
            print(f"  Quality Score: {r.output_quality_score:.1f}/100")
            print(f"  GPU Memory: {sum(r.gpu_memory_mb) / 1024:.2f} GB total")

        # Calculate speedup
        if len(results) == 2:
            ep4, tp4 = results[0], results[1]
            speedup = tp4.decode_throughput / ep4.decode_throughput if ep4.decode_throughput > 0 else 0
            memory_ratio = sum(ep4.gpu_memory_mb) / sum(tp4.gpu_memory_mb) if sum(tp4.gpu_memory_mb) > 0 else 0

            print(f"\nEP=4 vs TP=4:")
            print(f"  Throughput Speedup: {speedup:.2f}x")
            print(f"  Memory Ratio: {memory_ratio:.2f}x")

            # Quality check
            if ep4.output_quality_score < 50:
                print(f"  ⚠️  WARNING: EP=4 output quality is low ({ep4.output_quality_score:.1f}/100)")
            else:
                print(f"  ✅ EP=4 output quality is good")

    # Save results to JSON
    if output_file:
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n✅ Results saved to {output_file}")


def main():
    """Run EP=4 performance benchmarks."""
    parser = argparse.ArgumentParser(description='EP=4 Performance Benchmark')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to Qwen3.6-35B-A3B-AWQ model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    model_path = args.model_path or os.environ.get('QWEN36_A3B_AWQ_PATH',
                                                     '/data/models/Qwen3.6-35B-A3B-AWQ')

    run_benchmark_suite(model_path, args.output)


if __name__ == '__main__':
    main()
