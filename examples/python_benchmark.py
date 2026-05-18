#!/usr/bin/env python3
"""
LMDeploy Python Benchmark Tool

Performs performance benchmarks using Python LMDeploy pipeline:
- Multiple context lengths (1K, 4K, 8K tokens)
- TTFT (Time To First Token) measurement using stream API
- Prefill and decode speed tracking
- GPU memory usage monitoring

Matches the Rust benchmark configuration for fair comparison.

Usage:
    python3 examples/python_benchmark.py [model_path]

Example:
    python3 examples/python_benchmark.py /mnt/eaget-4tb/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lmdeploy import Pipeline, TurbomindEngineConfig
from lmdeploy.messages import GenerationConfig


# Configuration matching Rust benchmark
CONTEXT_LENGTHS = [1024, 4096, 8192]
OUTPUT_LENGTH = 512
ITERATIONS = 3
WARMUP_ITERATIONS = 1
MODEL_PATH = "/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ"


class PythonBenchmark:
    """Benchmark runner using Python LMDeploy pipeline."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None
        self.tokenizer = None

    def initialize(self) -> float:
        """Initialize the pipeline and return initialization time in seconds."""
        print(f"Model: {self.model_path}")

        # Check model format
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            import json as json_lib
            with open(config_path) as f:
                config = json_lib.load(f)
                quant_method = config.get("quantization_config", {}).get("quant_method", None)
                if quant_method == "awq":
                    print("Detected AWQ quantization, configuring quant_policy=4")
                    os.environ["LMDEPLOY_QUANT_POLICY"] = "4"

        start_time = time.perf_counter()

        # Create TurboMind engine config
        engine_config = TurbomindEngineConfig(
            tp=1,
            cache_max_entry_count=0.8,
            session_len=8192 * 2,  # Allow some headroom
            max_batch_size=1,
        )

        # Initialize pipeline
        self.pipeline = Pipeline(
            self.model_path,
            backend_config=engine_config,
            log_level="WARNING",
        )

        # Get tokenizer for accurate token counting
        self.tokenizer = self.pipeline.async_engine.tokenizer

        init_time = time.perf_counter() - start_time
        print(f"Pipeline initialized in {init_time:.2f}s\n")

        return init_time

    def generate_prompt(self, target_tokens: int) -> str:
        """Generate a prompt of approximately target_tokens length."""
        # Sample text, roughly 4 chars per token
        sample_text = "The quick brown fox jumps over the lazy dog. "
        char_length = target_tokens * 4
        repeats = (char_length // len(sample_text)) + 1
        return sample_text * repeats

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if self.tokenizer is None:
            # Rough estimate: 4 chars per token
            return len(text) // 4
        return len(self.tokenizer.encode(text))

    def get_gpu_memory_mb(self) -> float:
        """Get GPU memory usage in MB using nvidia-smi."""
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip().split("\n")[0])
        except Exception:
            return 0.0

    def run_stream_benchmark(
        self,
        prompt: str,
        max_tokens: int,
        stream: bool = True,
    ) -> dict:
        """Run benchmark with stream API for accurate TTFT measurement."""
        prompt_tokens = self.count_tokens(prompt)

        if stream:
            return self._run_stream_inference(prompt, max_tokens, prompt_tokens)
        else:
            return self._run_non_stream_inference(prompt, max_tokens, prompt_tokens)

    def _run_stream_inference(
        self,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int,
    ) -> dict:
        """Run stream inference and measure TTFT accurately."""
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=0.1,  # Low temp for deterministic output
            top_p=1.0,
            top_k=1,
        )

        start_time = time.perf_counter()
        ttft_time = None
        output_text = ""
        decode_start = None

        # Use stream_infer for accurate TTFT measurement
        # Returns an iterator - we need to iterate to get streaming responses
        stream_gen = self.pipeline.stream_infer(
            prompt,
            gen_config=gen_config,
            stream_response=True,
        )

        # stream_gen yields Response objects (or iterators of them for batches)
        for item in stream_gen:
            # For single prompt, item can be:
            # 1. A Response object (final response)
            # 2. An iterator yielding Response objects (streaming chunks)

            if hasattr(item, 'text'):
                # Direct Response object (non-streaming or final result)
                response = item
                if ttft_time is None:
                    ttft_time = time.perf_counter() - start_time
                if response.text:
                    output_text = response.text
            else:
                # Iterator of Response objects (streaming)
                try:
                    for response in item:
                        if ttft_time is None and response.text:
                            ttft_time = time.perf_counter() - start_time
                            decode_start = time.perf_counter()
                        if response.text:
                            output_text = response.text  # Accumulate (or use extend)
                except TypeError:
                    # item might be a single Response object that's not iterable
                    if hasattr(item, 'text') and item.text:
                        if ttft_time is None:
                            ttft_time = time.perf_counter() - start_time
                        output_text = item.text

        end_time = time.perf_counter()

        total_time = end_time - start_time
        decode_time = end_time - decode_start if decode_start is not None else total_time - (ttft_time if ttft_time else total_time * 0.1)
        output_tokens = self.count_tokens(output_text)

        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_time * 1000 if ttft_time else 0,
            "prefill_time_ms": (ttft_time * 1000) if ttft_time else (total_time * 100),
            "decode_time_ms": decode_time * 1000,
            "total_time_ms": total_time * 1000,
            "prefill_speed_tps": (prompt_tokens / ttft_time) if ttft_time and ttft_time > 0 else 0,
            "decode_speed_tps": (output_tokens / decode_time) if decode_time > 0 else 0,
        }

    def _run_non_stream_inference(
        self,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int,
    ) -> dict:
        """Run non-stream inference."""
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=1.0,
            top_k=1,
        )

        start_time = time.perf_counter()
        responses = self.pipeline.infer(prompt, gen_config=gen_config)
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # responses is a list of Response objects
        if isinstance(responses, list) and len(responses) > 0:
            response = responses[0]
        else:
            response = responses

        output_text = getattr(response, "text", "")
        output_tokens = self.count_tokens(output_text)

        # Estimate TTFT as ~20% of total time (typical for non-stream)
        ttft_ms = total_time * 200  # 20% in ms
        decode_time = total_time - (ttft_ms / 1000)

        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "prefill_time_ms": ttft_ms,
            "decode_time_ms": decode_time * 1000,
            "total_time_ms": total_time * 1000,
            "prefill_speed_tps": (prompt_tokens / (ttft_ms / 1000)) if ttft_ms > 0 else 0,
            "decode_speed_tps": (output_tokens / decode_time) if decode_time > 0 else 0,
        }

    def run_single_benchmark(
        self,
        context_length: int,
        output_length: int,
        iteration: int,
        stream: bool = True,
    ) -> dict:
        """Run a single benchmark iteration."""
        prompt = self.generate_prompt(context_length)
        prompt_tokens = self.count_tokens(prompt)

        print(f"  Run {iteration}: context={context_length}, prompt_tokens={prompt_tokens}", end=" ", flush=True)

        result = self.run_stream_benchmark(prompt, output_length, stream=stream)

        print(
            f"TTFT={result['ttft_ms']:.0f}ms, "
            f"Prefill={result['prefill_speed_tps']:.0f} t/s, "
            f"Decode={result['decode_speed_tps']:.0f} t/s"
        )

        result["context_length"] = context_length
        result["iteration"] = iteration
        result["output_length"] = result["output_tokens"]  # Actual output tokens

        return result

    def run_full_benchmark(self, stream: bool = True) -> dict:
        """Run the complete benchmark suite."""
        print("=" * 80)
        print("LMDeploy Python Performance Benchmark")
        print("=" * 80)
        print(f"Context lengths: {[f'{l}K' if l >= 1024 else str(l) for l in CONTEXT_LENGTHS]}")
        print(f"Output length: {OUTPUT_LENGTH} tokens")
        print(f"Iterations: {ITERATIONS}")
        print(f"Stream mode: {stream}")
        print("=" * 80)

        all_results = []
        initial_memory = self.get_gpu_memory_mb()

        for context_length in CONTEXT_LENGTHS:
            print(f"\n### Context {context_length} tokens ###")

            # Warmup runs
            for i in range(WARMUP_ITERATIONS):
                print(f"  Warmup {i + 1}/{WARMUP_ITERATIONS}...", end=" ", flush=True)
                self.run_single_benchmark(context_length, OUTPUT_LENGTH, 0, stream=stream)
                time.sleep(0.5)  # Small pause between runs

            # Measured runs
            for iter_num in range(1, ITERATIONS + 1):
                result = self.run_single_benchmark(context_length, OUTPUT_LENGTH, iter_num, stream=stream)
                all_results.append(result)

        # Calculate summaries
        summaries = self._calculate_summaries(all_results)

        # Get final memory
        final_memory = self.get_gpu_memory_mb()
        memory_used_mb = final_memory - initial_memory

        # Build report
        report = {
            "engine": "Python LMDeploy Pipeline",
            "model": self.model_path,
            "gpu_memory_used_mb": memory_used_mb,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "context_lengths": CONTEXT_LENGTHS,
                "output_length": OUTPUT_LENGTH,
                "iterations": ITERATIONS,
                "warmup_iterations": WARMUP_ITERATIONS,
                "stream": stream,
            },
            "all_results": all_results,
            "summaries": summaries,
        }

        return report

    def _calculate_summaries(self, results: list) -> list:
        """Calculate summary statistics by context length."""
        summaries = []

        for context_length in CONTEXT_LENGTHS:
            context_results = [r for r in results if r["context_length"] == context_length]
            if not context_results:
                continue

            ttfts = [r["ttft_ms"] for r in context_results]
            prefills = [r["prefill_speed_tps"] for r in context_results]
            decodes = [r["decode_speed_tps"] for r in context_results]
            totals = [r["total_time_ms"] for r in context_results]

            summaries.append({
                "context_length": context_length,
                "output_length": OUTPUT_LENGTH,
                "iterations": len(context_results),
                "avg_ttft_ms": float(np.mean(ttfts)),
                "min_ttft_ms": float(np.min(ttfts)),
                "max_ttft_ms": float(np.max(ttfts)),
                "avg_prefill_speed_tps": float(np.mean(prefills)),
                "avg_decode_speed_tps": float(np.mean(decodes)),
                "avg_total_time_ms": float(np.mean(totals)),
            })

        return summaries

    def print_summary(self, report: dict):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("Benchmark Summary")
        print("=" * 80)
        print(f"GPU Memory Used: {report['gpu_memory_used_mb'] / 1024:.2f} GB")
        print()

        for summary in report["summaries"]:
            ctx_label = f"{summary['context_length'] // 1024}K" if summary["context_length"] >= 1024 else str(summary["context_length"])
            print(f"{ctx_label} context:")
            print(f"  Avg TTFT:        {summary['avg_ttft_ms']:.2f} ms")
            print(f"  Avg Prefill:     {summary['avg_prefill_speed_tps']:.2f} tokens/s")
            print(f"  Avg Decode:      {summary['avg_decode_speed_tps']:.2f} tokens/s")
            print(f"  Avg Total Time:  {summary['avg_total_time_ms']:.2f} ms")
            print()

    def save_results(self, report: dict, filename: str | None = None) -> str:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"python_benchmark_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Results saved to: {filename}")
        return filename


def main():
    """Main entry point."""
    # Parse model path from CLI
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    print(f"\n=== LMDeploy Python Benchmark ===")
    print(f"Using model: {model_path}\n")

    # Initialize benchmark
    benchmark = PythonBenchmark(model_path)

    # Initialize pipeline and measure time
    print("Initializing pipeline...")
    init_time = benchmark.initialize()
    print(f"Initialization complete in {init_time:.2f}s\n")

    # Run benchmarks
    # Try stream mode first, fall back to non-stream if it fails
    stream_mode = True
    try:
        report = benchmark.run_full_benchmark(stream=stream_mode)
    except Exception as e:
        print(f"\nStream mode failed ({e}), falling back to non-stream mode...")
        stream_mode = False
        report = benchmark.run_full_benchmark(stream=stream_mode)

    # Print summary
    benchmark.print_summary(report)

    # Save results
    benchmark.save_results(report)

    # Clean up
    print("\nClosing pipeline...")
    benchmark.pipeline.close()

    print("\n=== Benchmark Complete ===")
    return report


if __name__ == "__main__":
    main()