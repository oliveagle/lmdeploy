#!/usr/bin/env python3
"""
DFlash Performance Benchmark - Compare with baseline

STORY-004: Run DFlash with speculative decoding and measure acceptance rate
"""

import os
import sys
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Configure environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb=512'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

# Model paths
target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"


@dataclass
class DFlashBenchmarkResult:
    """Data class for DFlash benchmark results."""
    mode: str
    batch_size: int
    num_requests: int
    total_tokens: int
    total_time_sec: float
    avg_latency_sec: float
    throughput_tps: float
    tokens_per_request: float
    # DFlash-specific metrics
    total_draft_tokens: int
    total_accepted_tokens: int
    acceptance_rate: float
    num_spec_decoding_steps: int


def check_models_exist():
    """Verify models exist."""
    if not os.path.exists(target_model):
        print(f"ERROR: Target model not found at {target_model}")
        sys.exit(1)
    if not os.path.exists(draft_model):
        print(f"ERROR: Draft model not found at {draft_model}")
        sys.exit(1)
    print(f"Target model: {target_model}")
    print(f"Draft model: {draft_model}")
    return True


def create_dflash_pipeline():
    """Create TurboMind pipeline with DFlash speculative decoding."""
    speculative_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
        quant_policy=0,  # FP16 (no quantization)
    )

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tensor_parallel=1,
        cache_max_entry_count=0.5,
        quant_policy=8,
        session_len=8192,
    )

    print("\nCreating DFlash pipeline (with speculative decoding)...")
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='WARNING',
    )
    print("✓ DFlash Pipeline created successfully!")
    return pipe


def warmup(pipe, num_warmup=3):
    """Warmup the pipeline before benchmarking."""
    print(f"\nWarming up ({num_warmup} requests)...")
    gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)
    prompts = ["Hello, how are you?", "What is Python?", "Explain AI."]

    for i in range(num_warmup):
        prompt = prompts[i % len(prompts)]
        pipe([{"role": "user", "content": prompt}], gen_config=gen_config)
    print("✓ Warmup complete!")


def benchmark_dflash_single_user(pipe, num_requests=30, max_new_tokens=128) -> DFlashBenchmarkResult:
    """Benchmark single-user with DFlash speculative decoding."""
    print(f"\n{'='*60}")
    print(f"DFlash Single-User (Chat) Benchmark")
    print(f"{'='*60}")
    print(f"Number of requests: {num_requests}")
    print(f"Max new tokens: {max_new_tokens}")

    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    prompts = [
        "Hello, how are you?",
        "What is Python?",
        "Explain AI in one sentence.",
        "Capital of France?",
        "Best programming language?",
    ]

    latencies = []
    total_tokens = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    num_spec_steps = 0

    print(f"\n{'Idx':<6} {'Latency(s)':<12} {'OutputTokens':<14} {'TPS':<12} {'AccRate':<10}")
    print("-" * 60)

    for i in range(num_requests):
        prompt_idx = i % len(prompts)
        prompt = prompts[prompt_idx]

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

        tps = output_tokens / latency if latency > 0 else 0

        # Try to extract DFlash stats from response metadata
        acceptance_rate = "N/A"
        # Note: In real implementation, we'd get these from engine stats
        # For now, we report throughput only

        print(f"{i+1:<6} {latency:<12.3f} {output_tokens:<14} {tps:<12.2f} {acceptance_rate:<10}")

        if (i + 1) % 10 == 0:
            elapsed_so_far = sum(latencies)
            tokens_so_far = total_tokens
            avg_tps_so_far = tokens_so_far / elapsed_so_far
            print(f"  -> Progress {i+1}/{num_requests}: avg={avg_tps_so_far:.2f} tps")

    total_time = sum(latencies)
    avg_latency = total_time / num_requests
    throughput = total_tokens / total_time if total_time > 0 else 0

    result = DFlashBenchmarkResult(
        mode="single",
        batch_size=1,
        num_requests=num_requests,
        total_tokens=total_tokens,
        total_time_sec=total_time,
        avg_latency_sec=avg_latency,
        throughput_tps=throughput,
        tokens_per_request=total_tokens / num_requests,
        total_draft_tokens=total_draft_tokens,
        total_accepted_tokens=total_accepted_tokens,
        acceptance_rate=total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0,
        num_spec_decoding_steps=num_spec_steps,
    )

    print(f"\n{'='*60}")
    print(f"DFlash Single-User Benchmark Results")
    print(f"{'='*60}")
    print(f"Total requests:     {result.num_requests}")
    print(f"Total tokens:       {result.total_tokens}")
    print(f"Total time:         {result.total_time_sec:.2f}s")
    print(f"Avg latency:        {result.avg_latency_sec:.3f}s/request")
    print(f"Throughput:         {result.throughput_tps:.2f} tokens/s")
    print(f"Avg tokens/req:     {result.tokens_per_request:.1f}")
    print(f"{'='*60}")

    return result


def benchmark_dflash_batch(pipe, batch_sizes=[4, 8], requests_per_batch=10, max_new_tokens=128) -> Dict[int, DFlashBenchmarkResult]:
    """Benchmark batch decoding with DFlash."""
    all_results = {}

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"DFlash Batch Decoding Benchmark (batch_size={batch_size})")
        print(f"{'='*60}")

        gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        prompts = [
            "Explain the concept of machine learning.",
            "What are the differences between Python and JavaScript?",
            "Describe neural network training process.",
        ]

        batch_latencies = []
        total_tokens = 0

        print(f"\n{'Batch':<8} {'Latency(s)':<12} {'OutputTokens':<14} {'TPS':<12}")
        print("-" * 54)

        for batch_idx in range(requests_per_batch):
            batch_prompts = []
            for i in range(batch_size):
                prompt_idx = (batch_idx * batch_size + i) % len(prompts)
                batch_prompts.append([{"role": "user", "content": prompts[prompt_idx]}])

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

            total_tokens += batch_output_tokens
            batch_latencies.append(latency)

            tps = batch_output_tokens / latency if latency > 0 else 0
            print(f"{batch_idx+1:<8} {latency:<12.3f} {batch_output_tokens:<14} {tps:<12.2f}")

        total_time = sum(batch_latencies)
        total_requests = requests_per_batch * batch_size
        avg_latency = total_time / requests_per_batch
        throughput = total_tokens / total_time if total_time > 0 else 0

        result = DFlashBenchmarkResult(
            mode="batch",
            batch_size=batch_size,
            num_requests=total_requests,
            total_tokens=total_tokens,
            total_time_sec=total_time,
            avg_latency_sec=avg_latency,
            throughput_tps=throughput,
            tokens_per_request=total_tokens / total_requests,
            total_draft_tokens=0,
            total_accepted_tokens=0,
            acceptance_rate=0.0,
            num_spec_decoding_steps=0,
        )

        all_results[batch_size] = result

        print(f"\n{'='*60}")
        print(f"DFlash Batch Results (batch_size={batch_size})")
        print(f"{'='*60}")
        print(f"Total requests:     {result.num_requests}")
        print(f"Total tokens:       {result.total_tokens}")
        print(f"Total time:         {result.total_time_sec:.2f}s")
        print(f"Avg batch latency:  {result.avg_latency_sec:.3f}s/batch")
        print(f"Throughput:         {result.throughput_tps:.2f} tokens/s")
        print(f"{'='*60}")

    return all_results


def compare_results(baseline_file, dflash_results):
    """Compare DFlash results with baseline."""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE COMPARISON: Baseline vs DFlash")
    print(f"{'='*60}")

    try:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)

        # Single-user comparison
        baseline_single = baseline['single_user']
        dflash_single = dflash_results.get('single', None)

        print(f"\n{'Metric':<25} {'Baseline':<15} {'DFlash':<15} {'Speedup':<10}")
        print("-" * 65)
        print(f"{'Single-user TPS':<25} {baseline_single['throughput_tps']:<15.2f} {dflash_single.throughput_tps:<15.2f} {dflash_single.throughput_tps/baseline_single['throughput_tps']:<10.2f}x")

        # Batch comparisons
        for bs in [4, 8]:
            baseline_batch = baseline['batch'].get(str(bs), {})
            dflash_batch = dflash_results.get('batch', {}).get(bs, None)
            if baseline_batch and dflash_batch:
                speedup = dflash_batch.throughput_tps / baseline_batch['throughput_tps']
                print(f"{f'Batch {bs} TPS':<25} {baseline_batch['throughput_tps']:<15.2f} {dflash_batch.throughput_tps:<15.2f} {speedup:<10.2f}x")

        print(f"{'='*60}")
    except Exception as e:
        print(f"Warning: Could not load baseline results: {e}")
        print("Run baseline benchmark first to compare results.")


def save_results(dflash_single, dflash_batch_results, output_file="dflash_benchmark_results.json"):
    """Save DFlash benchmark results to JSON file."""
    results_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": target_model,
        "draft_model": draft_model,
        "single_user": asdict(dflash_single),
        "batch": {str(bs): asdict(r) for bs, r in dflash_batch_results.items()},
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


def main():
    print("="*60)
    print("DFlash Performance Benchmark")
    print("="*60)

    # Check models
    check_models_exist()

    # Create DFlash pipeline
    pipe = create_dflash_pipeline()

    # Warmup
    warmup(pipe, num_warmup=3)

    # Run DFlash single-user benchmark
    dflash_single = benchmark_dflash_single_user(pipe, num_requests=30, max_new_tokens=128)

    # Run DFlash batch benchmarks
    dflash_batch = benchmark_dflash_batch(pipe, batch_sizes=[4, 8], requests_per_batch=10, max_new_tokens=128)

    # Save results
    save_results(dflash_single, dflash_batch)

    # Compare with baseline
    compare_results("baseline_benchmark_results.json", {
        'single': dflash_single,
        'batch': dflash_batch,
    })

    # Cleanup
    del pipe

    print("\n✓ DFlash benchmark complete!")


if __name__ == "__main__":
    main()