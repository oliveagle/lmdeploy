#!/usr/bin/env python3
"""
Performance Root Cause Analysis for LMDeploy

Analyzes three execution modes:
1. Python Direct: lmdeploy Python API → C++ library
2. Rust+C++ (PureCpp): Rust FFI → C++ library → gRPC
3. PyBridge: subprocess → Python → C++ library

Measures time breakdown at each layer to identify bottlenecks.
"""

import cProfile
import json
import os
import pstats
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/models/deepseek-llm-67b-chat")
TEST_CONTEXT_LENGTH = 512
TEST_OUTPUT_TOKENS = 64


def time_it(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


class Profiler:
    """Profile code execution and extract hotspots."""

    def __init__(self, name: str):
        self.name = name
        self.profiler = cProfile.Profile()
        self.stats: Optional[pstats.Stats] = None

    def __enter__(self):
        self.profiler.enable()
        return self

    def __exit__(self, *args):
        self.profiler.disable()
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        self.stats = ps
        self.output = s.getvalue()

    def get_hotspots(self) -> List[Dict[str, Any]]:
        """Extract top time-consuming functions."""
        if not self.stats:
            return []

        hotspots = []
        stats = self.stats.stats

        for (fn, (_, _, _, _, callers)) in sorted(
            stats.items(),
            key=lambda x: x[1][1].cumtime,  # Sort by cumulative time
            reverse=True
        )[:20]:
            filename, line, func_name = fn
            hotspots.append({
                "function": f"{func_name}:{line}",
                "file": Path(filename).name,
                "cumtime": callers.cumtime,
                "tottime": callers.tottime,
                "ncalls": callers.ncalls,
            })

        return hotspots


# ========================================
# Mode 1: Python Direct Analysis
# ========================================

def analyze_python_direct(
    model_path: str,
    context_length: int,
    output_tokens: int
) -> Dict[str, Any]:
    """
    Analyze Python Direct mode:
    - lmdeploy API overhead
    - Python → C++ FFI overhead
    - Actual C++ execution time

    Expected call chain:
    lmdeploy.turbomind.TurboMind.create_instance()
      → turbomind.py (Python wrapper)
        → _turbomind.so (pybind11)
          → libturbomind_c.so (C++ library)
            → CUDA kernels
    """
    from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
    from lmdeploy.turbomind import TurboMind
    from lmdeploy.tokenizer import Tokenizer

    result = {
        "mode": "Python Direct",
        "breakdown": {},
        "hotspots": [],
    }

    # 1. Measure model loading time
    @time_it
    def load_model():
        engine_config = TurbomindEngineConfig(
            session_len=32768,
            max_batch_size=32,
            cache_block_seq_len=64,
            tp=1,
        )
        return TurboMind(model_path=model_path, engine_config=engine_config)

    tm, load_time = load_model()
    result["breakdown"]["model_load_ms"] = load_time * 1000

    # 2. Measure tokenizer creation
    @time_it
    def load_tokenizer():
        return Tokenizer(model_path)

    tokenizer, tokenizer_time = load_tokenizer()
    result["breakdown"]["tokenizer_load_ms"] = tokenizer_time * 1000

    # 3. Measure instance creation
    @time_it
    def create_instance():
        return tm.create_instance()

    instance, instance_time = create_instance()
    result["breakdown"]["instance_create_ms"] = instance_time * 1000

    # 4. Profile the actual inference
    sample = "The quick brown fox jumps over the lazy dog. "
    prompt = sample * (context_length // len(sample) + 1)
    input_ids = tokenizer.encode(prompt)
    gen_config = GenerationConfig(max_new_tokens=output_tokens, temperature=0.7)

    with Profiler("python_inference") as prof:
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        for output in instance.stream_infer(
            session_id=0,
            input_ids=input_ids,
            gen_config=gen_config,
            sequence_start=True,
            sequence_end=True,
            stream_output=True,
        ):
            if output.status.value in (1, 2):
                elapsed = (time.perf_counter() - start) * 1000
                if first_token_time is None:
                    first_token_time = elapsed
                token_count += len(output.token_ids)

        total_time_ms = (time.perf_counter() - start) * 1000

    result["breakdown"]["ttft_ms"] = first_token_time or 0
    result["breakdown"]["total_inference_ms"] = total_time_ms
    result["breakdown"]["decode_ms"] = total_time_ms - (first_token_time or 0)
    result["breakdown"]["input_tokens"] = len(input_ids)
    result["breakdown"]["output_tokens"] = token_count
    result["hotspots"] = prof.get_hotspots()
    result["profile_output"] = prof.output

    tm.close()
    return result


# ========================================
# Mode 2: Rust PureCpp Analysis
# ========================================

def analyze_rust_purecpp(
    model_path: str,
    context_length: int,
    output_tokens: int
) -> Dict[str, Any]:
    """
    Analyze Rust+C++ mode:
    - gRPC client overhead
    - Network/IPC latency
    - Rust FFI → C++ overhead
    - Actual C++ execution time

    Expected call chain:
    Client (Python) → gRPC
      → Rust server (lmdeploy-rust-server)
        → FFI call to libturbomind_c.so
          → CUDA kernels
    """
    import grpc

    # Import gRPC stubs
    sys.path.insert(0, str(Path(__file__).parent / "tests"))
    import lmdeploy.v1.lm_deploy_pb2_grpc as lm_deploy_grpc
    import lmdeploy.v1.lm_deploy_pb2 as lm_deploy_pb2

    result = {
        "mode": "Rust PureCpp",
        "breakdown": {},
        "hotspots": [],
    }

    # 1. Measure connection time
    @time_it
    def connect():
        return grpc.insecure_channel("localhost:50051")

    channel, connect_time = connect()
    result["breakdown"]["connection_ms"] = connect_time * 1000

    stub = lm_deploy_grpc.LmDeployServiceStub(channel)

    # 2. Prepare request (measure serialization time)
    sample = "The quick brown fox jumps over the lazy dog. "
    prompt = sample * (context_length // len(sample) + 1)

    @time_it
    def serialize_request():
        return lm_deploy_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=output_tokens,
            temperature=0.7,
            top_p=0.95,
        )

    request, serialize_time = serialize_request()
    result["breakdown"]["request_serialize_ms"] = serialize_time * 1000

    # 3. Profile gRPC call
    with Profiler("grpc_call") as prof:
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        response_iterator = stub.generate_stream(request)

        for response in response_iterator:
            elapsed = (time.perf_counter() - start) * 1000
            if first_token_time is None and response.payload.chunk.text:
                first_token_time = elapsed
            if response.payload.chunk.text and response.payload.chunk.text != "[DONE]":
                token_count += 1

        total_time_ms = (time.perf_counter() - start) * 1000

    result["breakdown"]["ttft_ms"] = first_token_time or 0
    result["breakdown"]["total_inference_ms"] = total_time_ms
    result["breakdown"]["decode_ms"] = total_time_ms - (first_token_time or 0)
    result["breakdown"]["output_tokens"] = token_count
    result["hotspots"] = prof.get_hotspots()
    result["profile_output"] = prof.output

    channel.close()
    return result


# ========================================
# Mode 3: PyBridge Analysis
# ========================================

def analyze_pybridge(
    model_path: str,
    context_length: int,
    output_tokens: int
) -> Dict[str, Any]:
    """
    Analyze PyBridge mode:
    - Subprocess creation overhead
    - JSON serialization/deserialization
    - Inter-process communication (IPC) overhead
    - Python → C++ FFI overhead
    - Actual C++ execution time

    Expected call chain:
    Client (Python) → subprocess
      → Python bridge script (stdin/stdout)
        → lmdeploy API
          → _turbomind.so
            → libturbomind_c.so
              → CUDA kernels
    """
    result = {
        "mode": "PyBridge",
        "breakdown": {},
        "hotspots": [],
    }

    bridge_script = Path(__file__).parent.parent / "lmdeploy" / "turbomind" / "python_bridge.py"

    if not bridge_script.exists():
        result["error"] = f"Bridge script not found: {bridge_script}"
        return result

    # 1. Measure subprocess startup
    @time_it
    def start_bridge():
        return subprocess.Popen(
            ["python3", str(bridge_script), "--model-path", model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    proc, startup_time = start_bridge()
    result["breakdown"]["subprocess_startup_ms"] = startup_time * 1000

    # 2. Wait for ready response
    ready_start = time.perf_counter()
    ready_line = proc.stdout.readline()
    ready_time = (time.perf_counter() - ready_start) * 1000
    ready = json.loads(ready_line)
    result["breakdown"]["ready_wait_ms"] = ready_time

    if ready.get("status") != "ok":
        result["error"] = f"Bridge failed: {ready}"
        return result

    # 3. Tokenize locally
    from lmdeploy.tokenizer import Tokenizer
    tokenizer = Tokenizer(model_path)
    sample = "The quick brown fox jumps over the lazy dog. "
    prompt = sample * (context_length // len(sample) + 1)
    input_ids = tokenizer.encode(prompt)

    # 4. Measure JSON serialization
    cmd = {
        "cmd": "generate_stream",
        "input_ids": input_ids,
        "max_new_tokens": output_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
    }

    @time_it
    def serialize_command():
        return json.dumps(cmd) + "\n"

    cmd_str, serialize_time = serialize_command()
    result["breakdown"]["json_serialize_ms"] = serialize_time * 1000

    # 5. Profile the streaming response
    with Profiler("pybridge_stream") as prof:
        start = time.perf_counter()
        first_token_time = None
        token_count = 0
        done = False

        while not done:
            line_start = time.perf_counter()
            line = proc.stdout.readline().strip()
            if not line:
                break

            response = json.loads(line)
            if response.get("status") == "ok":
                if response.get("type") == "done":
                    done = True
                else:
                    elapsed = (time.perf_counter() - start) * 1000
                    if first_token_time is None:
                        first_token_time = elapsed
                    token_count += 1

        total_time_ms = (time.perf_counter() - start) * 1000

    result["breakdown"]["ttft_ms"] = first_token_time or 0
    result["breakdown"]["total_inference_ms"] = total_time_ms
    result["breakdown"]["decode_ms"] = total_time_ms - (first_token_time or 0)
    result["breakdown"]["input_tokens"] = len(input_ids)
    result["breakdown"]["output_tokens"] = token_count
    result["hotspots"] = prof.get_hotspots()
    result["profile_output"] = prof.output

    # Cleanup
    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=10)

    return result


# ========================================
# Comparative Analysis
# ========================================

def compare_modes(
    python_result: Dict[str, Any],
    rust_result: Dict[str, Any],
    pybridge_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare the three modes and identify bottlenecks.
    """
    comparison = {
        "relative_performance": {},
        "bottleneck_analysis": {},
        "recommendations": [],
    }

    # Calculate overhead percentages
    for mode_result, mode_name in [
        (python_result, "python_direct"),
        (rust_result, "rust_purecpp"),
        (pybridge_result, "pybridge"),
    ]:
        if "error" in mode_result:
            continue

        total = mode_result["breakdown"]["total_inference_ms"]
        ttft = mode_result["breakdown"]["ttft_ms"]
        decode = mode_result["breakdown"]["decode_ms"]

        comparison["relative_performance"][mode_name] = {
            "total_ms": total,
            "ttft_ms": ttft,
            "decode_ms": decode,
            "ttft_percentage": (ttft / total * 100) if total > 0 else 0,
            "decode_percentage": (decode / total * 100) if total > 0 else 0,
        }

    # Analyze bottlenecks
    if "error" not in python_result:
        # Python Direct baseline
        baseline_total = python_result["breakdown"]["total_inference_ms"]

        if "error" not in rust_result:
            rust_overhead = (
                (rust_result["breakdown"]["total_inference_ms"] - baseline_total)
                / baseline_total * 100
            )
            comparison["bottleneck_analysis"]["rust_overhead_percent"] = rust_overhead

        if "error" not in pybridge_result:
            pybridge_overhead = (
                (pybridge_result["breakdown"]["total_inference_ms"] - baseline_total)
                / baseline_total * 100
            )
            comparison["bottleneck_analysis"]["pybridge_overhead_percent"] = pybridge_overhead

    # Generate recommendations based on hotspots
    for mode_result, mode_name in [
        (python_result, "Python Direct"),
        (rust_result, "Rust PureCpp"),
        (pybridge_result, "PyBridge"),
    ]:
        if "error" in mode_result:
            continue

        # Check for Python overhead
        python_hotspots = [
            h for h in mode_result.get("hotspots", [])
            if h.get("file", "").endswith(".py")
        ]
        if python_hotspots:
            comparison["recommendations"].append({
                "mode": mode_name,
                "issue": f"Python overhead detected in {len(python_hotspots)} functions",
                "suggestion": "Consider moving hot paths to compiled code",
            })

    return comparison


def print_analysis_report(
    python_result: Dict[str, Any],
    rust_result: Dict[str, Any],
    pybridge_result: Dict[str, Any],
    comparison: Dict[str, Any],
):
    """Print detailed analysis report."""
    print("=" * 80)
    print("PERFORMANCE ROOT CAUSE ANALYSIS REPORT")
    print("=" * 80)

    for mode_result in [python_result, rust_result, pybridge_result]:
        mode_name = mode_result.get("mode", "Unknown")

        if "error" in mode_result:
            print(f"\n{mode_name}: ERROR - {mode_result['error']}")
            continue

        print(f"\n{'─' * 80}")
        print(f"Mode: {mode_name}")
        print(f"{'─' * 80}")

        breakdown = mode_result.get("breakdown", {})

        print("\nTiming Breakdown:")
        for key, value in sorted(breakdown.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.2f} ms")
            else:
                print(f"  {key}: {value}")

        # Calculate derived metrics
        if "input_tokens" in breakdown and "output_tokens" in breakdown:
            input_toks = breakdown["input_tokens"]
            output_toks = breakdown["output_tokens"]
            ttft = breakdown.get("ttft_ms", 0)
            decode_time = breakdown.get("decode_ms", 1)

            prefill_tps = (input_toks / ttft * 1000) if ttft > 0 else 0
            decode_tps = (output_toks / decode_time * 1000) if decode_time > 0 else 0

            print(f"\nDerived Metrics:")
            print(f"  Prefill speed: {prefill_tps:.1f} tokens/sec")
            print(f"  Decode speed: {decode_tps:.1f} tokens/sec")

        # Show hotspots
        hotspots = mode_result.get("hotspots", [])[:5]
        if hotspots:
            print(f"\nTop 5 Hotspots:")
            for i, h in enumerate(hotspots, 1):
                print(f"  {i}. {h['function']} ({h['cumtime']:.2f}s)")

    # Print comparison
    print(f"\n{'=' * 80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'=' * 80}")

    rel_perf = comparison.get("relative_performance", {})
    if rel_perf:
        print("\nRelative Performance:")
        for mode_name, metrics in rel_perf.items():
            print(f"\n  {mode_name}:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.2f}")

    bottlenecks = comparison.get("bottleneck_analysis", {})
    if bottlenecks:
        print("\nBottleneck Analysis:")
        for key, value in bottlenecks.items():
            print(f"  {key}: {value:.2f}%")

    recommendations = comparison.get("recommendations", [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - [{rec['mode']}] {rec['issue']}")
            print(f"    → {rec['suggestion']}")


def main():
    """Run full root cause analysis."""
    print("Running Performance Root Cause Analysis...")
    print(f"Model: {MODEL_PATH}")
    print(f"Context: {TEST_CONTEXT_LENGTH} tokens")
    print(f"Output: {TEST_OUTPUT_TOKENS} tokens")

    # Run analysis for each mode
    print("\n" + "=" * 60)
    print("Analyzing Python Direct mode...")
    python_result = analyze_python_direct(
        MODEL_PATH, TEST_CONTEXT_LENGTH, TEST_OUTPUT_TOKENS
    )

    print("\n" + "=" * 60)
    print("Analyzing Rust PureCpp mode...")
    rust_result = analyze_rust_purecpp(
        MODEL_PATH, TEST_CONTEXT_LENGTH, TEST_OUTPUT_TOKENS
    )

    print("\n" + "=" * 60)
    print("Analyzing PyBridge mode...")
    pybridge_result = analyze_pybridge(
        MODEL_PATH, TEST_CONTEXT_LENGTH, TEST_OUTPUT_TOKENS
    )

    # Compare modes
    comparison = compare_modes(python_result, rust_result, pybridge_result)

    # Print report
    print_analysis_report(python_result, rust_result, pybridge_result, comparison)

    # Save results
    output_file = Path(__file__).parent / "root_cause_analysis.json"
    with open(output_file, "w") as f:
        json.dump({
            "python_direct": python_result,
            "rust_purecpp": rust_result,
            "pybridge": pybridge_result,
            "comparison": comparison,
        }, f, indent=2)
    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
