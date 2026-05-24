#!/usr/bin/env python3
"""
Flame Graph Profiling for LMDeploy Performance Analysis

Generates flame graphs comparing:
1. Python Direct: Python API → C++ library
2. Rust+C++ (PureCpp): Rust FFI → C++ library

Requirements:
- py-spy: pip install py-spy (for Python profiling)
- perf: sudo apt-get install linux-tools-generic (for Rust profiling)
- flamegraph.pl: https://github.com/brendangregg/FlameGraph

Output:
- SVG flame graphs for each mode
- Collapsed stack traces for analysis
- Side-by-side comparison report
- Detailed bottleneck analysis

Usage:
    # Profile both Python and Rust
    python tests/flamegraph_profiler.py --model-path /path/to/model --mode both

    # Profile only Python
    python tests/flamegraph_profiler.py --mode python

    # Profile only Rust (requires running Rust server)
    python tests/flamegraph_profiler.py --mode rust --rust-pid <PID>
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


class FlameGraphProfiler:
    """Generate flame graphs for different LMDeploy execution modes."""

    def __init__(self, output_dir: Path, model_path: str, context_length: int = 512):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.context_length = context_length

        # Paths to external tools
        self.flamegraph_pl = Path("/tmp/flamegraph.pl")
        if not self.flamegraph_pl.exists():
            # Try to find it in common locations
            for path in [
                Path("/usr/local/bin/flamegraph.pl"),
                Path("/usr/bin/flamegraph.pl"),
                Path("~/.local/bin/flamegraph.pl"),
            ]:
                if path.expanduser().exists():
                    self.flamegraph_pl = path.expanduser()
                    break

        self.perf_available = self._check_perf()
        self.pyspy_available = self._check_pyspy()

    def _check_perf(self) -> bool:
        """Check if perf is available."""
        try:
            result = subprocess.run(
                ["perf", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_pyspy(self) -> bool:
        """Check if py-spy is available."""
        try:
            result = subprocess.run(
                ["py-spy", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def profile_python_direct(
        self,
        output_tokens: int = 64,
        duration_sec: int = 30,
    ) -> Dict[str, str]:
        """
        Profile Python Direct mode using py-spy.

        Generates:
        - python_direct.svg: Interactive flame graph
        - python_direct.collapsed: Collapsed stack traces
        """
        if not self.pyspy_available:
            return {"error": "py-spy not available. Install: pip install py-spy"}

        print(f"\n{'='*60}")
        print("Profiling Python Direct mode with py-spy...")
        print(f"{'='*60}")

        output_svg = self.output_dir / "python_direct.svg"
        output_collapsed = self.output_dir / "python_direct.collapsed"

        # Create a test script that will be profiled
        test_script = self._create_python_test_script(output_tokens)

        # Run py-spy
        cmd = [
            "py-spy",
            "record",
            "-o", str(output_svg),
            "--rate", "100",  # Sample at 100 Hz
            "--format", "flamegraph",
            "--",
            sys.executable,
            str(test_script),
        ]

        print(f"Running: {' '.join(cmd)}")
        print(f"Duration: {duration_sec} seconds")

        try:
            # py-spy runs for the duration of the program
            result = subprocess.run(
                cmd,
                timeout=duration_sec + 30,  # Extra time for startup
                env={**os.environ, "MODEL_PATH": self.model_path}
            )

            if result.returncode == 0 and output_svg.exists():
                # Also generate collapsed format for diffing
                self._svg_to_collapsed(output_svg, output_collapsed)

                return {
                    "svg": str(output_svg),
                    "collapsed": str(output_collapsed),
                    "status": "success"
                }
            else:
                return {
                    "error": f"py-spy failed with return code {result.returncode}",
                    "stderr": result.stderr.decode() if result.stderr else ""
                }

        except subprocess.TimeoutExpired:
            return {"error": "Profiling timed out"}
        except Exception as e:
            return {"error": str(e)}

    def profile_rust_purecpp(
        self,
        output_tokens: int = 64,
        duration_sec: int = 30,
        rust_pid: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Profile Rust+C++ mode using perf.

        Generates:
        - rust_purecpp.svg: Interactive flame graph
        - rust_purecpp.collapsed: Collapsed stack traces
        - rust_purecpp.perf: Raw perf data
        """
        if not self.perf_available:
            return {"error": "perf not available. Install: sudo apt-get install linux-tools-generic"}

        print(f"\n{'='*60}")
        print("Profiling Rust+C++ mode with perf...")
        print(f"{'='*60}")

        output_svg = self.output_dir / "rust_purecpp.svg"
        output_collapsed = self.output_dir / "rust_purecpp.collapsed"
        output_perf = self.output_dir / "rust_purecpp.perf"

        # Find the Rust server process or use provided PID
        if rust_pid is None:
            rust_pid = self._find_rust_server()

        if rust_pid is None:
            return {"error": "Rust server not running. Start lmdeploy-rust-server first or provide --rust-pid."}

        print(f"Profiling Rust process PID: {rust_pid}")

        # Run perf record
        perf_data = self.output_dir / "rust_perf.data"
        cmd = [
            "perf",
            "record",
            "-F", "99",  # Sample at 99 Hz
            "-p", str(rust_pid),
            "-o", str(perf_data),
            "--call-graph", "dwarf",  # Use DWARF for accurate stack traces
            "--sleep", str(duration_sec * 1000),  # Duration in milliseconds
        ]

        print(f"Running: {' '.join(cmd)}")
        print(f"Duration: {duration_sec} seconds")

        try:
            # Run perf record
            result = subprocess.run(
                cmd,
                timeout=duration_sec + 30,
                env={"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}
            )

            if result.returncode != 0:
                return {
                    "error": f"perf record failed with return code {result.returncode}",
                    "stderr": result.stderr.decode() if result.stderr else ""
                }

            # Convert to collapsed format
            collapse_cmd = [
                "perf",
                "script",
                "-i", str(perf_data),
                "--no-inline"
            ]

            collapse_result = subprocess.run(
                collapse_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if collapse_result.returncode == 0:
                # Parse and collapse the output
                collapsed = self._collapse_perf_script(collapse_result.stdout)

                with open(output_collapsed, "w") as f:
                    f.write(collapsed)

                # Generate SVG if flamegraph.pl is available
                if self.flamegraph_pl.exists():
                    svg_result = subprocess.run(
                        ["perl", str(self.flamegraph_pl)],
                        input=collapsed,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if svg_result.returncode == 0:
                        with open(output_svg, "w") as f:
                            f.write(svg_result.stdout)
                    else:
                        return {"error": "flamegraph.pl failed to generate SVG"}

                # Save raw perf script output
                with open(output_perf, "w") as f:
                    f.write(collapse_result.stdout)

                # Clean up perf data
                perf_data.unlink(missing_ok=True)

                return {
                    "svg": str(output_svg),
                    "collapsed": str(output_collapsed),
                    "perf_script": str(output_perf),
                    "status": "success"
                }
            else:
                return {
                    "error": f"perf script failed: {collapse_result.stderr}"
                }

        except subprocess.TimeoutExpired:
            return {"error": "Profiling timed out"}
        except Exception as e:
            return {"error": str(e)}

    def _find_rust_server(self) -> Optional[int]:
        """Find the running Rust server process."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "lmdeploy-rust-server"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                return int(pids[0])

        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    def _collapse_perf_script(self, perf_script: str) -> str:
        """
        Collapse perf script output to flamegraph format.

        Input format:
            lmdeploy-rust-ser 12345 [000] 12345.678::
                ffff800000000000 __libc_start_main
                ffff800000000000 main
                ffff800000000000 rust_function

        Output format:
            __libc_start_main;main;rust_function 1
        """
        collapsed_lines = []
        current_stack = []

        for line in perf_script.split("\n"):
            line = line.strip()

            # Skip empty lines and headers
            if not line or line.startswith("#") or "/" in line:
                if current_stack:
                    # End of a stack trace
                    if len(current_stack) > 1:
                        collapsed_lines.append(";".join(reversed(current_stack)) + " 1")
                    current_stack = []
                continue

            # Parse stack frame
            # Format: address function_name
            parts = line.split()
            if len(parts) >= 2:
                # The function name is the last part
                func_name = parts[-1]

                # Filter out kernel frames and noise
                if not any(x in func_name for x in ["[kernel]", "kallsyms", "unknown"]):
                    # Simplify C++ names
                    func_name = self._simplify_function_name(func_name)
                    current_stack.append(func_name)

        return "\n".join(collapsed_lines)

    def _simplify_function_name(self, name: str) -> str:
        """Simplify C++/Rust function names for flame graph readability."""
        # Remove template parameters
        if "<" in name and ">" in name:
            # Keep base name, remove template
            name = name.split("<")[0] + ">"

        # Remove namespace prefixes that are too long
        prefixes = [
            "turbomind::",
            "lmdeploy_rust_server::",
            "std::",
            "alloc::",
            "core::",
        ]

        for prefix in prefixes:
            if name.startswith(prefix):
                # Keep one level of namespace
                parts = prefix.split("::")
                if len(parts) > 1:
                    name = parts[-1] + name[len(prefix):]

        # Truncate very long names
        if len(name) > 80:
            name = name[:77] + "..."

        return name

    def _svg_to_collapsed(self, svg_path: Path, output_path: Path):
        """Extract stack information from SVG to collapsed format."""
        # This is a simplified version - real parsing would need XML parsing
        # For now, we'll create a placeholder
        output_path.write_text("# Flame graph data (use SVG for visualization)\n")

    def _create_python_test_script(self, output_tokens: int) -> Path:
        """Create a standalone Python script for profiling."""
        script_content = f'''#!/usr/bin/env python3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL_PATH = "{self.model_path}"
CONTEXT_LENGTH = {self.context_length}
OUTPUT_TOKENS = {output_tokens}

def main():
    print("Loading model...")
    engine_config = TurbomindEngineConfig(
        session_len=32768,
        max_batch_size=32,
        cache_block_seq_len=64,
        tp=1,
    )
    tm = TurboMind(model_path=MODEL_PATH, engine_config=engine_config)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    print("Creating instance...")
    instance = tm.create_instance()

    print("Running inference...")
    sample = "The quick brown fox jumps over the lazy dog. "
    prompt = sample * (CONTEXT_LENGTH // len(sample) + 1)
    input_ids = tokenizer.encode(prompt)

    gen_config = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.7)

    start = time.time()
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
            token_count += len(output.token_ids)

    elapsed = time.time() - start
    print(f"Generated {{token_count}} tokens in {{elapsed:.2f}}s")
    print(f"Throughput: {{token_count / elapsed:.1f}} tokens/sec")

    tm.close()

if __name__ == "__main__":
    main()
'''

        script_path = self.output_dir / "profile_python_direct.py"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        return script_path

    def generate_diff_report(self) -> Dict[str, Any]:
        """
        Generate a differential report comparing the two flame graphs.

        Identifies:
        - Functions unique to each mode
        - Functions with significantly different time spent
        - Stack depth differences
        """
        python_collapsed = self.output_dir / "python_direct.collapsed"
        rust_collapsed = self.output_dir / "rust_purecpp.collapsed"

        if not python_collapsed.exists() or not rust_collapsed.exists():
            return {"error": "Collapsed files not found. Run profiling first."}

        # Parse collapsed files
        python_stacks = self._parse_collapsed(python_collapsed)
        rust_stacks = self._parse_collapsed(rust_collapsed)

        # Compare
        report = {
            "python_only": [],
            "rust_only": [],
            "common_functions": {},
            "stack_depth": {
                "python": self._max_stack_depth(python_stacks),
                "rust": self._max_stack_depth(rust_stacks),
            },
            "total_samples": {
                "python": sum(python_stacks.values()),
                "rust": sum(rust_stacks.values()),
            }
        }

        # Find unique and common functions
        python_funcs = set(self._extract_functions(python_stacks))
        rust_funcs = set(self._extract_functions(rust_stacks))

        report["python_only"] = sorted(python_funcs - rust_funcs)
        report["rust_only"] = sorted(rust_funcs - python_funcs)

        # For common functions, compare sample counts
        common = python_funcs & rust_funcs
        for func in common:
            python_count = sum(
                count for stack, count in python_stacks.items()
                if func in stack
            )
            rust_count = sum(
                count for stack, count in rust_stacks.items()
                if func in stack
            )
            report["common_functions"][func] = {
                "python_samples": python_count,
                "rust_samples": rust_count,
                "ratio": rust_count / python_count if python_count > 0 else float('inf')
            }

        return report

    def _parse_collapsed(self, path: Path) -> Dict[str, int]:
        """Parse a collapsed stack file."""
        stacks = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    stack, count = parts
                    stacks[stack] = int(count)

        return stacks

    def _extract_functions(self, stacks: Dict[str, int]) -> List[str]:
        """Extract all unique function names from stacks."""
        functions = set()
        for stack in stacks.keys():
            functions.update(stack.split(";"))
        return list(functions)

    def _max_stack_depth(self, stacks: Dict[str, int]) -> int:
        """Find maximum stack depth."""
        return max((len(stack.split(";")) for stack in stacks.keys()), default=0)

    def analyze_bottlenecks(self) -> Dict:
        """
        Analyze specific bottlenecks in Rust+C++ vs Python+C++ call stacks.

        Focuses on:
        - FFI overhead (Rust→C++ boundary crossings)
        - Memory allocation patterns
        - Synchronization primitives (locks, semaphores)
        - Token callback chain
        - Prefill/decode pipeline differences
        """
        python_collapsed = self.output_dir / "python_direct.collapsed"
        rust_collapsed = self.output_dir / "rust_purecpp.collapsed"

        if not python_collapsed.exists() and not rust_collapsed.exists():
            return {"error": "No collapsed files found. Run profiling first."}

        analysis = {
            "ffi_boundary": {},
            "memory_allocation": {},
            "synchronization": {},
            "token_processing": {},
            "top_functions": {"python": [], "rust": []},
        }

        # Patterns to look for
        ffi_patterns = {
            "ffi_boundary": ["ctypes", "cffi", "extern", "PyCFunction", "libturbomind", "TurboMindC"],
            "memory_allocation": ["malloc", "free", "alloc", "dealloc", "mmap", "cudaMalloc", "cudaMemcpy"],
            "synchronization": ["lock", "unlock", "mutex", "semaphore", "wait", "sleep", "parking_lot"],
            "token_processing": ["token_callback", "decode", "forward", "generate", "infer"],
        }

        # Parse and analyze each mode
        for mode, collapsed_path in [("python", python_collapsed), ("rust", rust_collapsed)]:
            if not collapsed_path.exists():
                continue

            stacks = self._parse_collapsed(collapsed_path)
            if not stacks:
                continue

            # Get top 20 functions by sample count
            func_counts = {}
            for stack, count in stacks.items():
                for func in stack.split(";"):
                    func_counts[func] = func_counts.get(func, 0) + count

            sorted_funcs = sorted(func_counts.items(), key=lambda x: x[1], reverse=True)
            analysis["top_functions"][mode] = [{"name": name, "samples": count, "pct": count / sum(func_counts.values()) * 100} for name, count in sorted_funcs[:20]]

            # Analyze pattern groups
            for pattern_name, patterns in ffi_patterns.items():
                pattern_samples = sum(
                    count for stack, count in stacks.items()
                    if any(p in stack.lower() for p in patterns)
                )
                total_samples = sum(stacks.values())
                analysis[pattern_name][mode] = {
                    "total_samples": pattern_samples,
                    "percentage": pattern_samples / total_samples * 100 if total_samples > 0 else 0,
                }

        # Calculate bottlenecks
        for category in ["ffi_boundary", "memory_allocation", "synchronization", "token_processing"]:
            if "python" in analysis[category] and "rust" in analysis[category]:
                rust_pct = analysis[category]["rust"]["percentage"]
                python_pct = analysis[category]["python"]["percentage"]
                diff = rust_pct - python_pct

                analysis[category]["bottleneck"] = {
                    "rust_overhead_pct": diff,
                    "is_significant": abs(diff) > 5.0,
                    "description": self._get_bottleneck_description(category, diff),
                }

        return analysis

    def _get_bottleneck_description(self, category: str, diff: float) -> str:
        """Get human-readable description of a bottleneck."""
        descriptions = {
            "ffi_boundary": {
                "high": "Rust has significantly higher FFI boundary overhead than Python. Check FFI call frequency and parameter marshaling.",
                "low": "Rust has lower FFI overhead than Python. FFI boundary is well optimized.",
                "normal": "FFI overhead is similar between Rust and Python.",
            },
            "memory_allocation": {
                "high": "Rust has higher memory allocation overhead. Check for excessive Vec/String allocations in hot paths.",
                "low": "Rust has better memory efficiency than Python.",
                "normal": "Memory allocation patterns are similar.",
            },
            "synchronization": {
                "high": "Rust has more synchronization overhead. Check for lock contention in spawn_blocking, request pool, or async channels.",
                "low": "Rust has lower synchronization overhead.",
                "normal": "Synchronization patterns are similar.",
            },
            "token_processing": {
                "high": "Rust token processing is slower. Check token_callback for allocation and decode overhead.",
                "low": "Rust token processing is faster.",
                "normal": "Token processing is similar.",
            },
        }

        if abs(diff) > 10:
            key = "high" if diff > 0 else "low"
        elif abs(diff) > 2:
            key = "high" if diff > 0 else "low"
        else:
            key = "normal"

        return descriptions.get(category, {}).get(key, "Unknown")

    def generate_callstack_report(self) -> str:
        """
        Generate a detailed call stack comparison report.
        """
        analysis = self.analyze_bottlenecks()

        if "error" in analysis:
            return f"Error: {analysis['error']}"

        lines = []
        lines.append("=" * 70)
        lines.append("FLAME GRAPH BOTTLENECK ANALYSIS REPORT")
        lines.append("Rust+C++ vs Python+C++ Call Stack Comparison")
        lines.append("=" * 70)

        # Top functions comparison
        lines.append("\n## Top Functions by Sample Count")
        lines.append("")

        for mode in ["python", "rust"]:
            top_funcs = analysis["top_functions"][mode]
            if not top_funcs:
                continue

            lines.append(f"\n### {mode.upper()} Top 20 Functions")
            lines.append(f"{'Rank':<6}{'Function':<40}{'Samples':<12}{'%':<10}")
            lines.append("-" * 68)

            for i, func in enumerate(top_funcs, 1):
                lines.append(f"{i:<6}{func['name']:<40}{func['samples']:<12}{func['pct']:.1f}%")

        # Bottleneck analysis
        lines.append("\n## Bottleneck Analysis")
        lines.append("")

        for category in ["ffi_boundary", "memory_allocation", "synchronization", "token_processing"]:
            if "python" in analysis[category] and "rust" in analysis[category]:
                bottleneck = analysis[category].get("bottleneck", {})
                rust_pct = analysis[category]["rust"]["percentage"]
                python_pct = analysis[category]["python"]["percentage"]

                lines.append(f"\n### {category.replace('_', ' ').title()}")
                lines.append(f"Python: {python_pct:.1f}% | Rust: {rust_pct:.1f}% | Diff: {bottleneck.get('rust_overhead_pct', 0):.1f}%")
                lines.append(f"Significant: {'YES' if bottleneck.get('is_significant') else 'No'}")
                lines.append(f"Analysis: {bottleneck.get('description', 'N/A')}")

        return "\n".join(lines)


def print_report(profiler: FlameGraphProfiler):
    """Print a formatted profiling report."""
    report = profiler.generate_diff_report()

    if "error" in report:
        print(f"\nError: {report['error']}")
        return

    print("\n" + "="*60)
    print("FLAME GRAPH COMPARISON REPORT")
    print("="*60)

    # Stack depth
    print("\nStack Depth:")
    print(f"  Python Direct: {report['stack_depth']['python']} frames")
    print(f"  Rust+C++:      {report['stack_depth']['rust']} frames")

    # Total samples
    print("\nTotal Samples:")
    print(f"  Python Direct: {report['total_samples']['python']}")
    print(f"  Rust+C++:      {report['total_samples']['rust']}")

    # Unique functions
    print("\nUnique Functions:")
    print(f"  Python only: {len(report['python_only'])} functions")
    if report['python_only']:
        for func in report['python_only'][:5]:
            print(f"    - {func}")
        if len(report['python_only']) > 5:
            print(f"    ... and {len(report['python_only']) - 5} more")

    print(f"  Rust only: {len(report['rust_only'])} functions")
    if report['rust_only']:
        for func in report['rust_only'][:5]:
            print(f"    - {func}")
        if len(report['rust_only']) > 5:
            print(f"    ... and {len(report['rust_only']) - 5} more")

    # Common functions with significant differences
    print("\nFunction Time Comparison (common functions):")
    print("  Function               | Python Samples | Rust Samples | Ratio")
    print("  " + "-"*70)

    common_sorted = sorted(
        report['common_functions'].items(),
        key=lambda x: x[1]['rust_samples'],
        reverse=True
    )[:10]

    for func, stats in common_sorted:
        print(f"  {func[:20]:20} | {stats['python_samples']:14d} | {stats['rust_samples']:12d} | {stats['ratio']:5.2f}x")

    print("\nOutput Files:")
    print(f"  {profiler.output_dir}/python_direct.svg")
    print(f"  {profiler.output_dir}/rust_purecpp.svg")
    print(f"  Open SVG files in a browser for interactive visualization")


def print_bottleneck_report(profiler: FlameGraphProfiler):
    """Print detailed bottleneck analysis report."""
    report = profiler.generate_callstack_report()
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="Generate flame graphs for LMDeploy performance analysis"
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/mnt/models/deepseek-llm-67b-chat"),
        help="Path to the model directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./flamegraph_results",
        help="Output directory for flame graphs"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Input context length in tokens"
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=64,
        help="Number of output tokens to generate"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Profiling duration in seconds"
    )
    parser.add_argument(
        "--mode",
        choices=["python", "rust", "both"],
        default="both",
        help="Which mode to profile"
    )
    parser.add_argument(
        "--rust-pid",
        type=int,
        default=None,
        help="PID of the Rust server process (if not auto-detected)"
    )
    parser.add_argument(
        "--analyze-bottlenecks",
        action="store_true",
        help="Generate detailed bottleneck analysis report"
    )

    args = parser.parse_args()

    profiler = FlameGraphProfiler(
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        context_length=args.context_length
    )

    print("LMDeploy Flame Graph Profiler")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Context: {args.context_length} tokens")
    print(f"Output: {args.output_tokens} tokens")
    print(f"Output dir: {args.output_dir}")

    # Check tools
    print("\nTool availability:")
    print(f"  py-spy: {'✓' if profiler.pyspy_available else '✗'}")
    print(f"  perf:   {'✓' if profiler.perf_available else '✗'}")
    print(f"  flamegraph.pl: {'✓' if profiler.flamegraph_pl.exists() else '✗'}")

    results = {}

    if args.mode in ["python", "both"]:
        results["python"] = profiler.profile_python_direct(
            output_tokens=args.output_tokens,
            duration_sec=args.duration
        )

    if args.mode in ["rust", "both"]:
        results["rust"] = profiler.profile_rust_purecpp(
            output_tokens=args.output_tokens,
            duration_sec=args.duration,
            rust_pid=args.rust_pid
        )

    # Save results summary
    summary_path = Path(args.output_dir) / "profiling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison report
    if args.mode == "both":
        print_report(profiler)

    # Print bottleneck analysis if requested
    if args.analyze_bottlenecks:
        print_bottleneck_report(profiler)
        # Also save to file
        bottleneck_report_path = Path(args.output_dir) / "bottleneck_analysis.md"
        report = profiler.generate_callstack_report()
        with open(bottleneck_report_path, "w") as f:
            f.write(report)
        print(f"\nBottleneck report saved to: {bottleneck_report_path}")

    print(f"\nResults saved to {args.output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
