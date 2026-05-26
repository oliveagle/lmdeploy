#!/usr/bin/env python3
"""
Unified Fair Benchmark Framework for Python vs Rust LMDeploy

This module provides a standardized interface for running identical performance comparisons
between Python TurboMind and Rust LMDeploy servers.

Features:
- Identical configuration for both engines
- Standardized measurement methodology
- Comprehensive metrics (TTFT, prefill TPS, decode TPS)
- Statistical analysis (min, max, avg, std dev)
- Comparable JSON output format
- Built-in warmup strategies
- Session isolation for fair measurement
"""

import os
import sys
import time
import json
import asyncio
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

# Configure logging
os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EngineType(Enum):
    """Type of LMDeploy engine."""
    PYTHON_TURBOMIND = "python_turbomind"
    RUST_GRPC = "rust_grpc"
    RUST_CPP_ENGINE = "rust_cpp_engine"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model_path: str
    test_cases: List[tuple] = field(default_factory=lambda: [
        ("1K", 1024),
        ("2K", 2048),
        ("4K", 4096),
        ("8K", 8192),
    ])
    warmup_iterations: int = 2
    measurement_iterations: int = 3
    max_new_tokens: int = 1
    stream_output: bool = True
    enable_prefix_caching: bool = False
    session_len: int = 16384
    cache_max_entry_count: float = 0.4
    tp: int = 1
    max_batch_size: int = 1
    model_format: str = "awq"
    repeat_text: str = "The quick brown fox jumps over the lazy dog. "


@dataclass
class SingleRunMetrics:
    """Metrics from a single benchmark run."""
    ttft_ms: float
    total_tokens: int
    elapsed_ms: float
    prefill_tps: float


@dataclass
class TestCaseMetrics:
    """Aggregated metrics for a single test case."""
    label: str
    target_tokens: int
    actual_tokens: int
    ttft_ms_list: List[float]
    ttft_avg_ms: float
    ttft_min_ms: float
    ttft_max_ms: float
    ttft_std_ms: float
    prefill_tps_list: List[float]
    prefill_tps_avg: float
    prefill_tps_min: float
    prefill_tps_max: float
    prefill_tps_std: float

    @classmethod
    def from_runs(cls, label: str, target_tokens: int, actual_tokens: int,
                    runs: List[SingleRunMetrics]) -> 'TestCaseMetrics':
        """Create aggregated metrics from a list of runs."""
        ttfts = [r.ttft_ms for r in runs]
        tps_list = [r.prefill_tps for r in runs]

        return cls(
            label=label,
            target_tokens=target_tokens,
            actual_tokens=actual_tokens,
            ttft_ms_list=ttfts,
            ttft_avg_ms=statistics.mean(ttfts),
            ttft_min_ms=min(ttfts),
            ttft_max_ms=max(ttfts),
            ttft_std_ms=statistics.stdev(ttfts) if len(ttfts) >= 2 else 0.0,
            prefill_tps_list=tps_list,
            prefill_tps_avg=statistics.mean(tps_list),
            prefill_tps_min=min(tps_list),
            prefill_tps_max=max(tps_list),
            prefill_tps_std=statistics.stdev(tps_list) if len(tps_list) >= 2 else 0.0,
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    engine_type: str
    engine_name: str
    model_path: str
    config: Dict[str, Any]
    timestamp: float
    test_cases: Dict[str, TestCaseMetrics]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "engine_type": self.engine_type,
            "engine_name": self.engine_name,
            "model_path": self.model_path,
            "config": self.config,
            "timestamp": self.timestamp,
            "test_cases": {k: asdict(v) for k, v in self.test_cases.items()},
        }

    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class BaseBenchmarkEngine(ABC):
    """Abstract base class for benchmark engines."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._tokenizer = None

    def _gen_prompt(self, token_count: int) -> str:
        """Generate a prompt of approximately the given token count."""
        repeats = max(1, token_count // 10)
        return self.config.repeat_text * repeats

    def generate_prompt(self, token_count: int) -> str:
        """Generate a prompt for benchmarking."""
        return self._gen_prompt(token_count)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine (load model, etc.)."""
        pass

    @abstractmethod
    async def warmup(self) -> None:
        """Run warmup iterations."""
        pass

    @abstractmethod
    async def run_single_test(self, label: str, target_tokens: int) -> SingleRunMetrics:
        """Run a single test iteration and return metrics."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""
        pass

    async def run_benchmark(self) -> BenchmarkResult:
        """Run the full benchmark suite."""
        await self.initialize()
        await self.warmup()

        test_case_metrics: Dict[str, TestCaseMetrics] = {}

        for label, target_tokens in self.config.test_cases:
            print(f"\n  Running {label} ({target_tokens} tokens)...")
            runs: List[SingleRunMetrics] = []
            actual_tokens = 0

            for i in range(self.config.measurement_iterations):
                run = await self.run_single_test(label, target_tokens)
                runs.append(run)
                actual_tokens = run.total_tokens
                print(f"    Iteration {i+1}: TTFT={run.ttft_ms:.1f}ms, "
                      f"Prefill={run.prefill_tps:.0f} tok/s")

            metrics = TestCaseMetrics.from_runs(
                label, target_tokens, actual_tokens, runs)
            test_case_metrics[label] = metrics

            print(f"  {label} Avg: TTFT={metrics.ttft_avg_ms:.1f}ms, "
                  f"Prefill={metrics.prefill_tps_avg:.0f} tok/s")

        await self.shutdown()

        result = BenchmarkResult(
            engine_type=self.__class__.__name__,
            engine_name=self.get_engine_name(),
            model_path=self.config.model_path,
            config=asdict(self.config),
            timestamp=time.time(),
            test_cases=test_case_metrics,
        )

        return result

    @abstractmethod
    def get_engine_name(self) -> str:
        """Return a human-readable name for this engine."""
        pass


class PythonTurboMindEngine(BaseBenchmarkEngine):
    """Benchmark engine for Python TurboMind."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._tm = None
        self._tokenizer = None
        self._instance = None
        self._session_counter = 10000  # Start higher to avoid collisions

    def _get_unique_session_id(self) -> int:
        """Get a unique session ID for each test run."""
        session_id = self._session_counter
        self._session_counter += 1000  # Large increment to avoid collisions
        return session_id

    async def initialize(self) -> None:
        """Initialize Python TurboMind engine."""
        from lmdeploy.turbomind import TurboMind
        from lmdeploy.tokenizer import Tokenizer
        from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig

        print("\n[Python TurboMind] Initializing...")
        self._tm = TurboMind(
            self.config.model_path,
            engine_config=TurbomindEngineConfig(
                session_len=self.config.session_len,
                max_batch_size=self.config.max_batch_size,
                tp=self.config.tp,
                model_format=self.config.model_format,
                cache_max_entry_count=self.config.cache_max_entry_count,
                enable_prefix_caching=self.config.enable_prefix_caching,
            )
        )
        self._tokenizer = Tokenizer(self.config.model_path)
        print("  OK: Python TurboMind initialized")

    async def warmup(self) -> None:
        """Run warmup iterations."""
        from lmdeploy.messages import GenerationConfig

        print("\n[Python TurboMind] Warmup...")
        warmup_prompt = self.generate_prompt(512)
        warmup_ids = self._tokenizer.encode(warmup_prompt)

        for _ in range(self.config.warmup_iterations):
            session_id = self._get_unique_session_id()
            instance = self._tm.create_instance()
            async for _ in instance.async_stream_infer(
                session_id=session_id,
                input_ids=warmup_ids,
                gen_config=GenerationConfig(max_new_tokens=1),
                sequence_start=True,
                sequence_end=True,
            ):
                pass

        print("  OK: Warmup complete")

    async def run_single_test(self, label: str, target_tokens: int) -> SingleRunMetrics:
        """Run a single test iteration."""
        from lmdeploy.messages import GenerationConfig

        prompt = self.generate_prompt(target_tokens)
        input_ids = self._tokenizer.encode(prompt)
        actual_tokens = len(input_ids)

        session_id = self._get_unique_session_id()
        instance = self._tm.create_instance()

        start = time.perf_counter()
        ttft = None

        async for output in instance.async_stream_infer(
            session_id=session_id,
            input_ids=input_ids,
            gen_config=GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.0,
            ),
            sequence_start=True,
            sequence_end=True,
            stream_output=True,
        ):
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000.0
                break

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        prefill_tps = actual_tokens / (ttft / 1000.0) if ttft > 0 else 0.0

        return SingleRunMetrics(
            ttft_ms=ttft,
            total_tokens=actual_tokens,
            elapsed_ms=elapsed_ms,
            prefill_tps=prefill_tps,
        )

    async def shutdown(self) -> None:
        """Shutdown Python TurboMind engine."""
        if self._tm is not None:
            self._tm.close()
            self._tm = None
            self._instance = None

    def get_engine_name(self) -> str:
        return "Python TurboMind"


class RustGrpcEngine(BaseBenchmarkEngine):
    """Benchmark engine for Rust LMDeploy gRPC server."""

    def __init__(self, config: BenchmarkConfig, host: str = "localhost", port: int = 50051):
        super().__init__(config)
        self._host = host
        self._port = port
        self._channel = None
        self._stub = None
        self._tokenizer = None
        self._session_counter = 1000

    async def initialize(self) -> None:
        """Initialize connection to Rust gRPC server."""
        import grpc
        try:
            from lmdeploy.v1 import lm_deploy_pb2
            from lmdeploy.v1 import lm_deploy_pb2_grpc
        except ImportError:
            lm_deploy_pb2 = None
            lm_deploy_pb2_grpc = None

        if lm_deploy_pb2 is None:
            raise RuntimeError("gRPC protobuf modules not available")

        print("\n[Rust gRPC] Initializing...")

        options = [
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
        ]

        self._channel = grpc.aio.insecure_channel(f"{self._host}:{self._port}", options=options)
        self._stub = lm_deploy_pb2_grpc.LmDeployServiceStub(self._channel)

        # Load tokenizer for prompt generation
        from lmdeploy.tokenizer import Tokenizer
        self._tokenizer = Tokenizer(self.config.model_path)

        print("  OK: Rust gRPC connection initialized")

    async def warmup(self) -> None:
        """Run warmup iterations."""
        from lmdeploy.v1 import lm_deploy_pb2

        print("\n[Rust gRPC] Warmup...")
        warmup_prompt = self.generate_prompt(512)

        for i in range(self.config.warmup_iterations):
            session_id = str(self._session_counter)
            self._session_counter += 1

            request = lm_deploy_pb2.GenerateRequest(
                session_id=session_id,
                prompt=warmup_prompt,
                max_new_tokens=1,
                temperature=0.0,
            )

            async for _ in self._stub.GenerateStream(request):
                pass

        print("  OK: Warmup complete")

    async def run_single_test(self, label: str, target_tokens: int) -> SingleRunMetrics:
        """Run a single test iteration."""
        from lmdeploy.v1 import lm_deploy_pb2

        prompt = self.generate_prompt(target_tokens)
        input_ids = self._tokenizer.encode(prompt)
        actual_tokens = len(input_ids)

        session_id = str(self._session_counter)
        self._session_counter += 1

        start = time.perf_counter()
        ttft = None

        request = lm_deploy_pb2.GenerateRequest(
            session_id=session_id,
            prompt=prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
        )

        async for response in self._stub.GenerateStream(request):
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000.0
                break

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        prefill_tps = actual_tokens / (ttft / 1000.0) if ttft > 0 else 0.0

        return SingleRunMetrics(
            ttft_ms=ttft,
            total_tokens=actual_tokens,
            elapsed_ms=elapsed_ms,
            prefill_tps=prefill_tps,
        )

    async def shutdown(self) -> None:
        """Shutdown gRPC connection."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    def get_engine_name(self) -> str:
        return "Rust LMDeploy gRPC"


class BenchmarkComparator:
    """Compare benchmark results from multiple engines."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results

    def print_comparison(self) -> None:
        """Print comparison table."""
        if not self.results:
            print("No results to compare")
            return

        test_labels = list(self.results[0].test_cases.keys())

        print("\n" + "=" * 120)
        print("PERFORMANCE COMPARISON".center(120))
        print("=" * 120)

        for label in test_labels:
            print(f"\n{label} Context Length".center(120))
            print("-" * 120)
            print(f"{'Engine':<30} {'TTFT Avg':>12} {'TTFT Min':>12} {'TTFT Max':>12} "
                  f"{'Prefill TPS':>15} {'Prefill Max':>15}")
            print("-" * 120)

            best_tps = 0.0
            best_engine = ""
            for result in self.results:
                if label in result.test_cases:
                    tc = result.test_cases[label]
                    print(f"{result.engine_name:<30} "
                          f"{tc.ttft_avg_ms:>10.1f}ms "
                          f"{tc.ttft_min_ms:>10.1f}ms "
                          f"{tc.ttft_max_ms:>10.1f}ms "
                          f"{tc.prefill_tps_avg:>13.0f} "
                          f"{tc.prefill_tps_max:>13.0f}")

                    if tc.prefill_tps_avg > best_tps:
                        best_tps = tc.prefill_tps_avg
                        best_engine = result.engine_name

            if len(self.results) >= 2:
                print("\n  Relative Performance:")
                base = self.results[0]
                if label in base.test_cases:
                    base_tps = base.test_cases[label].prefill_tps_avg
                    for result in self.results[1:]:
                        if label in result.test_cases:
                            curr_tps = result.test_cases[label].prefill_tps_avg
                            ratio = curr_tps / base_tps if base_tps > 0 else 0.0
                            print(f"    {result.engine_name} vs {base.engine_name}: "
                                  f"{ratio:.1%}")

        print("\n" + "=" * 120)

    def save_comparison(self, path: Path) -> None:
        """Save comparison to JSON file."""
        output = {
            "timestamp": time.time(),
            "results": [r.to_dict() for r in self.results],
        }

        with open(path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


async def run_benchmarks(config: BenchmarkConfig,
                        engines: Optional[List[EngineType]] = None) -> List[BenchmarkResult]:
    """Run benchmarks for multiple engines."""
    if engines is None:
        engines = [EngineType.PYTHON_TURBOMIND]

    results = []

    for engine_type in engines:
        print("\n" + "=" * 80)
        print(f"RUNNING BENCHMARK: {engine_type.value}")
        print("=" * 80)

        try:
            engine: Optional[BaseBenchmarkEngine] = None

            if engine_type == EngineType.PYTHON_TURBOMIND:
                engine = PythonTurboMindEngine(config)
            elif engine_type == EngineType.RUST_GRPC:
                engine = RustGrpcEngine(config)
            else:
                print(f"  SKIP: Unknown engine type {engine_type}")
                continue

            result = await engine.run_benchmark()
            results.append(result)

        except Exception as e:
            print(f"  ERROR: Benchmark failed for {engine_type.value}: {e}")
            import traceback
            traceback.print_exc()

    return results


def load_result(path: Path) -> Optional[BenchmarkResult]:
    """Load a benchmark result from JSON file."""
    if not path.exists():
        return None

    with open(path, 'r') as f:
        data = json.load(f)

    test_cases = {}
    for label, tc_data in data.get('test_cases', {}).items():
        test_cases[label] = TestCaseMetrics(**tc_data)

    return BenchmarkResult(
        engine_type=data.get('engine_type', ''),
        engine_name=data.get('engine_name', ''),
        model_path=data.get('model_path', ''),
        config=data.get('config', {}),
        timestamp=data.get('timestamp', 0.0),
        test_cases=test_cases,
    )
