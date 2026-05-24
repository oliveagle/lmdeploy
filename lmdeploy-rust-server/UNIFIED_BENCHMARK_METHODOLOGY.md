# Unified Benchmark Methodology - Python vs Rust Throughput

## Overview

This document defines the measurement methodology for comparing Python vs Rust TurboMind performance across all dimensions requested in the bead.

## Test Matrix

### Fixed Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| Model | Qwen3.6-35B-A3B-AWQ | Same model for all modes |
| GPU | Tesla V100 32GB | Same hardware |
| session_len | 32768 | Same max context window |
| max_batch_size | 32 | Same scheduler config |
| cache_block_seq_len | 64 | Same KV cache block size |
| tp (tensor parallel) | 1 | Single GPU |
| enable_prefix_caching | false | Disabled for clean measurements |
| temperature | 0.7 | Same generation config |

### Context Lengths

| Length | Label | Expected Pre-fill (Python) | Expected Pre-fill (Rust) |
|--------|-------|----------------------------|-------------------------|
| 512 tokens | 512 | ~40,000+ tok/s | ~1,700+ tok/s (before fix) |
| 2048 tokens | 2K | ~35,000+ tok/s | ~1,700+ tok/s (before fix) |
| 4096 tokens | 4K | ~30,000+ tok/s | ~1,700+ tok/s (before fix) |
| 8192 tokens | 8K | ~25,000+ tok/s | ~1,700+ tok/s (before fix) |
| 16384 tokens | 16K | ~20,000+ tok/s | ~1,700+ tok/s (before fix) |

### Run Configuration

- Warmup runs: 1 (per context length, not counted)
- Measured runs: 3 (per context length, averaged)

## Measurement Dimensions

### 1. Pure C++ Engine Call Time (不含 tokenization/pool)

**Python Direct:**
```python
# In test_three_modes.py
start = time.perf_counter()
for output in instance.stream_infer(...):
    # Measure from first token callback
    if first_token_time is None:
        first_token_time = (time.perf_counter() - start) * 1000
```

**Rust PureCpp:**
```rust
// In benchmark.rs
let start = Instant::now();
let mut stream = engine.generate_stream(&prompt, params).await;
// Measure from first token
while let Some(_token) = stream.next().await {
    if !first_token_received {
        ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
        first_token_received = true;
    }
}
```

### 2. Tokenization Time

**Python:**
```python
tok_start = time.perf_counter()
input_ids = tokenizer.encode(prompt)
tok_time = (time.perf_counter() - tok_start) * 1000  # milliseconds
```

**Rust:**
```rust
let tok_start = Instant::now();
let input_ids = tokenizer.encode(&prompt, false, false)?;
let tok_time = tok_start.elapsed().as_secs_f64() * 1000.0;
```

### 3. Pool Acquisition Time

**Rust:**
```rust
let pool_start = Instant::now();
let request = engine.pool.acquire().await;
let pool_time = pool_start.elapsed().as_secs_f64() * 1000.0;
```

**Python:**
```python
# Python TurboMind doesn't have explicit pool acquisition
# Engine instance is created once and reused
pool_time = 0  # N/A for Python Direct
```

### 4. End-to-End TTFT (Time To First Token)

**Definition:** Wall clock time from API call start to first token received.

**Python:** `stream_infer()` start → first yield
**Rust:** `generate_stream()` start → first `Some(token)` from stream

### 5. Prefill Speed

**Formula:** `prefill_speed_tps = input_tokens / (ttft_ms / 1000.0)`

- `input_tokens` = actual token count from tokenizer (not character count)
- `ttft_ms` = time from start to first token

### 6. Decode Speed

**Formula:** `decode_speed_tps = output_tokens / (decode_time_ms / 1000.0)`

- `output_tokens` = total tokens generated
- `decode_time_ms` = total_time_ms - ttft_ms

## Test Execution

### Python Benchmark

```bash
cd lmdeploy-rust-server/tests
MODEL_PATH=/path/to/qwen3.5-moe python test_three_modes.py
```

### Rust Benchmark

```bash
cd lmdeploy-rust-server
cargo run --example benchmark -- /path/to/model --quick --output-length 512
```

### Root Cause Profiling

```bash
cd lmdeploy-rust-server/tests
python root_cause_analysis.py
python flamegraph_profiler.py
```

## Files

| File | Purpose |
|------|---------|
| `tests/test_three_modes.py` | 3-mode comparison: Python Direct / PyBridge / Rust PureCpp |
| `tests/benchmark_comparison.py` | Context length sweep benchmark |
| `tests/root_cause_analysis.py` | cProfile hotspot analysis |
| `tests/flamegraph_profiler.py` | py-spy + perf flamegraph generation |
| `examples/benchmark.rs` | Rust native benchmark CLI tool |
| `RUST_CPP_BOTTLENECK_ANALYSIS.md` | 5 core bottlenecks documented |
| `docs/performance_root_cause_analysis.md` | English analysis document |

## Root Cause Summary

### 5 Identified Bottlenecks

1. **Polling vs Callback**: Rust uses `sleep(1ms)` + polling (1000 checks/sec) vs Python's event-driven `await sem.acquire()`
2. **spawn_blocking Thread Pool**: All Rust work goes through blocking thread pool, creating contention
3. **Request Pool Lock Contention**: `available_permits()` calculation race + `blocking_lock()` secondary wait
4. **token_callback Allocation**: New `Vec<u32>` allocated per token in C callback
5. **sync forward vs async forward**: Non-streaming uses efficient `forward()`, streaming uses `forward_async()` + polling

### Expected Performance After Fixes

| Metric | Before | After (Expected) | Improvement |
|--------|--------|-----------------|-------------|
| Prefill (8K) | ~1,727 tok/s | ~25,000+ tok/s | ~14x |
| CPU Usage (streaming) | High (polling) | Low (event-driven) | Significant |
| TTFT | High | Lower | Reduced polling latency |
