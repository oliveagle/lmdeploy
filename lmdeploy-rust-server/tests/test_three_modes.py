#!/usr/bin/env python3
"""
Rust vs Python TurboMind Performance Comparison

Tests three modes:
1. Pure C++ TurboMind (via Rust C FFI) - PureCpp
2. Python TurboMind API (via subprocess) - PyBridge
3. Python TurboMind API (direct) - Baseline

Measures:
- TTFT (Time To First Token)
- Prefill speed (tokens/sec)
- Decode speed (tokens/sec)
- Total latency
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer


# Test configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/models/deepseek-llm-67b-chat")
CONTEXT_LENGTHS = [512, 2048, 4096, 8192, 16384]
OUTPUT_TOKENS = 128
WARMUP_RUNS = 1
MEASURED_RUNS = 3


def generate_prompt(target_chars: int) -> str:
    """Generate a prompt of approximately target_chars characters."""
    sample = "The quick brown fox jumps over the lazy dog. "
    repeats = (target_chars // len(sample)) + 1
    return sample * repeats


async def test_python_turbomind_direct(
    model_path: str, context_length: int, output_tokens: int, iteration: int
) -> Dict[str, Any]:
    """Test Python TurboMind API directly (baseline)."""
    print(f"  [Python Direct] Context={context_length}, Iteration={iteration}")

    # Load engine
    engine_config = TurbomindEngineConfig(
        session_len=32768,
        max_batch_size=32,
        cache_block_seq_len=64,
        tp=1,
        enable_prefix_caching=False,
    )

    tm = TurboMind(model_path=model_path, engine_config=engine_config)
    tokenizer = Tokenizer(model_path)
    instance = tm.create_instance()

    # Generate prompt
    prompt = generate_prompt(context_length * 4)
    input_ids = tokenizer.encode(prompt)

    # Measure streaming generation
    gen_config = GenerationConfig(max_new_tokens=output_tokens, temperature=0.7)

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    output_ids = []

    for output in instance.stream_infer(
        session_id=iteration,
        input_ids=input_ids,
        gen_config=gen_config,
        sequence_start=True,
        sequence_end=True,
        stream_output=True,
    ):
        if output.status.value in (1, 2):  # SUCCESS or FINISH
            elapsed = (time.perf_counter() - start_time) * 1000
            if first_token_time is None:
                first_token_time = elapsed
            output_ids.extend(output.token_ids)
            token_count += len(output.token_ids)

    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Cleanup
    tm.close()

    return {
        "mode": "Python Direct",
        "context_length": len(input_ids),
        "output_tokens": token_count,
        "ttft_ms": first_token_time or 0,
        "total_time_ms": total_time_ms,
        "prefill_speed_tps": (len(input_ids) / (first_token_time or 1)) * 1000 if first_token_time else 0,
        "decode_speed_tps": (token_count / max(1, total_time_ms - (first_token_time or 0))) * 1000,
    }


async def test_python_bridge(
    model_path: str, context_length: int, output_tokens: int, iteration: int
) -> Dict[str, Any]:
    """Test Python Bridge (subprocess mode)."""
    print(f"  [PyBridge] Context={context_length}, Iteration={iteration}")

    # Start Python bridge subprocess
    bridge_script = Path(__file__).parent.parent / "lmdeploy" / "turbomind" / "python_bridge.py"
    proc = subprocess.Popen(
        ["python3", str(bridge_script), "--model-path", model_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for ready response
    ready_line = proc.stdout.readline()
    ready = json.loads(ready_line)
    if ready.get("status") != "ok":
        return {"error": f"Bridge failed to load: {ready}"}

    # Generate prompt
    prompt = generate_prompt(context_length * 4)

    # Tokenize locally (bridge doesn't have tokenizer exposed)
    sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))
    from lmdeploy.tokenizer import Tokenizer
    tokenizer = Tokenizer(model_path)
    input_ids = tokenizer.encode(prompt)

    # Send generate_stream command
    cmd = {
        "cmd": "generate_stream",
        "input_ids": input_ids,
        "max_new_tokens": output_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
    }

    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()

    # Collect streaming response
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    done = False

    while not done:
        line = proc.stdout.readline().strip()
        if not line:
            break

        response = json.loads(line)
        if response.get("status") == "ok":
            if response.get("type") == "done":
                done = True
            else:
                elapsed = (time.perf_counter() - start_time) * 1000
                if first_token_time is None:
                    first_token_time = elapsed
                token_count += 1

    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Shutdown
    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=10)

    return {
        "mode": "PyBridge",
        "context_length": len(input_ids),
        "output_tokens": token_count,
        "ttft_ms": first_token_time or 0,
        "total_time_ms": total_time_ms,
        "prefill_speed_tps": (len(input_ids) / (first_token_time or 1)) * 1000 if first_token_time else 0,
        "decode_speed_tps": (token_count / max(1, total_time_ms - (first_token_time or 0))) * 1000,
    }


async def test_rust_cpp_engine(
    model_path: str, context_length: int, output_tokens: int, iteration: int
) -> Dict[str, Any]:
    """Test Rust server with PureCpp engine."""
    print(f"  [Rust PureCpp] Context={context_length}, Iteration={iteration}")

    # Import grpc client
    sys.path.insert(0, str(Path(__file__).parent / "tests"))
    import grpc
    import lmdeploy.v1.lm_deploy_pb2_grpc as lm_deploy_grpc
    import lmdeploy.v1.lm_deploy_pb2 as lm_deploy_pb2

    # Connect to server
    channel = grpc.insecure_channel("localhost:50051")
    stub = lm_deploy_grpc.LmDeployServiceStub(channel)

    # Generate prompt
    prompt = generate_prompt(context_length * 4)

    # Call generate_stream
    request = lm_deploy_pb2.GenerateRequest(
        prompt=prompt,
        max_tokens=output_tokens,
        temperature=0.7,
        top_p=0.95,
    )

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        response_iterator = stub.generate_stream(request)

        for response in response_iterator:
            elapsed = (time.perf_counter() - start_time) * 1000
            if first_token_time is None and response.payload.chunk.text:
                first_token_time = elapsed
            if response.payload.chunk.text and response.payload.chunk.text != "[DONE]":
                token_count += 1

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "mode": "Rust PureCpp",
            "context_length": context_length,  # Approximate
            "output_tokens": token_count,
            "ttft_ms": first_token_time or 0,
            "total_time_ms": total_time_ms,
            "prefill_speed_tps": (context_length / (first_token_time or 1)) * 1000 if first_token_time else 0,
            "decode_speed_tps": (token_count / max(1, total_time_ms - (first_token_time or 0))) * 1000,
        }
    except grpc._channel._Rendezvous as e:
        return {"error": f"gRPC error: {e}"}
    finally:
        channel.close()


async def run_all_tests() -> Dict[str, List[Dict]]:
    """Run performance tests for all three modes."""
    results = {
        "python_direct": [],
        "pybridge": [],
        "rust_purecpp": [],
    }

    for context_len in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Testing Context Length: {context_len} tokens")
        print(f"{'='*60}")

        # Warmup runs
        for i in range(WARMUP_RUNS):
            try:
                await test_python_turbomind_direct(MODEL_PATH, context_len, OUTPUT_TOKENS, i)
            except Exception as e:
                print(f"  [Python Direct] Warmup error: {e}")

        # Measured runs
        for i in range(1, MEASURED_RUNS + 1):
            # Test Python Direct
            try:
                result = await test_python_turbomind_direct(
                    MODEL_PATH, context_len, OUTPUT_TOKENS, i
                )
                results["python_direct"].append(result)
                print(f"    TTFT: {result['ttft_ms']:.1f}ms, "
                      f"Prefill: {result['prefill_speed_tps']:.0f} tps, "
                      f"Decode: {result['decode_speed_tps']:.0f} tps")
            except Exception as e:
                print(f"  [Python Direct] Error: {e}")

            # Test PyBridge
            try:
                result = await test_python_bridge(MODEL_PATH, context_len, OUTPUT_TOKENS, i)
                results["pybridge"].append(result)
                print(f"    TTFT: {result['ttft_ms']:.1f}ms, "
                      f"Prefill: {result['prefill_speed_tps']:.0f} tps, "
                      f"Decode: {result['decode_speed_tps']:.0f} tps")
            except Exception as e:
                print(f"  [PyBridge] Error: {e}")

            # Test Rust PureCpp
            try:
                result = await test_rust_cpp_engine(MODEL_PATH, context_len, OUTPUT_TOKENS, i)
                if "error" not in result:
                    results["rust_purecpp"].append(result)
                    print(f"    TTFT: {result['ttft_ms']:.1f}ms, "
                          f"Prefill: {result['prefill_speed_tps']:.0f} tps, "
                          f"Decode: {result['decode_speed_tps']:.0f} tps")
                else:
                    print(f"  [Rust PureCpp] Error: {result['error']}")
            except Exception as e:
                print(f"  [Rust PureCpp] Error: {e}")

    return results


def print_summary(results: Dict[str, List[Dict]]):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    for mode, mode_results in results.items():
        if not mode_results or "error" in mode_results[0]:
            continue

        print(f"\n{mode.upper()}:")
        print(f"  {'Context':>10} | {'TTFT':>10} | {'Prefill TPS':>15} | {'Decode TPS':>15}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*15}-+-{'-'*15}")

        # Group by context length
        by_context: Dict[int, List[Dict]] = {}
        for r in mode_results:
            ctx = r["context_length"]
            if ctx not in by_context:
                by_context[ctx] = []
            by_context[ctx].append(r)

        for ctx in sorted(by_context.keys()):
            ctx_results = by_context[ctx]
            avg_ttft = sum(r["ttft_ms"] for r in ctx_results) / len(ctx_results)
            avg_prefill = sum(r["prefill_speed_tps"] for r in ctx_results) / len(ctx_results)
            avg_decode = sum(r["decode_speed_tps"] for r in ctx_results) / len(ctx_results)
            print(f"  {ctx:>10} | {avg_ttft:>10.1f} | {avg_prefill:>15.1f} | {avg_decode:>15.1f}")


async def main():
    """Main entry point."""
    print("Rust vs Python TurboMind Performance Comparison")
    print(f"Model: {MODEL_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print(f"Runs: {WARMUP_RUNS} warmup + {MEASURED_RUNS} measured")

    results = await run_all_tests()
    print_summary(results)

    # Save results
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
