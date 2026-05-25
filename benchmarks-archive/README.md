# Benchmarks Archive

This directory consolidates all benchmark-related code, results, and documentation for the LMDeploy project.

## Structure

```
benchmarks-archive/
├── scripts/      # Python benchmark scripts and test files
├── results/     # Benchmark result JSON files
├── docs/        # Benchmark documentation and reports
└── tools/       # Benchmark tools, binaries, and Go utilities
```

> **Note**: The `benchmark/` directory contains **active** benchmark code (official LMDeploy benchmarks).
> This archive contains **historical** and **experimental** benchmark files that were scattered across the project.

## Quick Reference

### Python Scripts (49 files)
| File | Description |
|------|-------------|
| `scripts/benchmark_main.py` | Main benchmark entry point |
| `scripts/benchmark_prefill.py` | Prefill performance testing |
| `scripts/benchmark_27b.py` | 27B model benchmark |
| `scripts/benchmark_27b_pytorch.py` | 27B PyTorch backend benchmark |
| `scripts/benchmark_tm.py` | TurboMind benchmark script |
| `scripts/benchmark_rust_prefill.py` | Rust server prefill benchmark |
| `scripts/benchmark_comparison.py` | Cross-backend comparison |
| `scripts/prefill_benchmark_*.py` | Various prefill test scripts |

### Results (12 files)
| Pattern | Description |
|---------|-------------|
| `results/benchmark_results_*.json` | Rust server benchmark results |
| `results/benchmark_python.py` | Python backend benchmark script |
| `results/benchmark_unified.py` | Unified benchmark script |

### Documentation (8 files)
| File | Description |
|------|-------------|
| `docs/BENCHMARK_RUST_VS_PYTHON_*.md` | Rust vs Python comparison reports |
| `docs/PRD_PERFORMANCE_BENCHMARK_*.md` | Performance benchmark PRD |
| `docs/benchmark_report_*.md` | Benchmark reports |

### Tools (8 files)
| File | Description |
|------|-------------|
| `tools/benchmark_tool_streaming.go` | Streaming benchmark tool (Go) |
| `tools/benchmark_tool.go` | Main benchmark tool (Go) |
| `tools/safetensors_benchmark` | Safetensors performance test binary |

### Rust Source Files (in `lmdeploy-rust-server/`)
```rust
lmdeploy-rust-server/benches/benchmark.rs          // Cargo benches
lmdeploy-rust-server/src/bin/prefill_benchmark.rs // Prefill benchmark binary
lmdeploy-rust-server/src/model/benchmark.rs       // Model benchmark utilities
lmdeploy-rust-server/examples/benchmark.rs        // Example benchmark code
```

### Scripts (in `lmdeploy-rust-server/`)
```bash
lmdeploy-rust-server/scripts/run_benchmark.sh      # Benchmark runner script
```

## Active vs Archived

| Directory | Contents | Status |
|-----------|----------|--------|
| `benchmark/` | Official LMDeploy benchmarks | Active |
| `benchmarks-archive/` | Historical/experimental benchmarks | Archived |
| `lmdeploy-rust-server/tests/` | Rust server integration tests | Active |

## History

Archived on 2026-05-26 to consolidate 77 scattered benchmark files from root level and various subdirectories.