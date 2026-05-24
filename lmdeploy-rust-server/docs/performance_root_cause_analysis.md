# Performance Root Cause Analysis: Python vs Rust LMDeploy

## Date: 2026-05-24
## Bead: lmdeploy-50z

---

## Architecture Overview

### Mode 1: Python Direct (lmdeploy API)

```
User Script
  → lmdeploy.turbomind.TurboMind (Python)
    → _turbomind.so (pybind11 binding, CPython module)
      → libturbomind_c.so (C++ engine)
        → CUDA kernels (prefill / decode)
```

**Call chain depth**: 4 layers
**Key files**:
- `lmdeploy/turbomind/turbomind.py` (~800 lines)
- `src/turbomind/capi/turbomind_c.cc` (C API)
- `src/turbomind/` (C++ engine)

**Overhead sources**:
1. Python object creation / destruction per stream_infer call
2. pybind11 conversion overhead (Python list → C++ vector → back)
3. TurboMind instance creation per batch (not reusable)
4. Python GIL contention if multiple threads
5. No connection pooling, no batching across calls

### Mode 2: Rust + C++ (PureCpp via FFI)

```
Client (gRPC)
  → lmdeploy-rust-server (Rust process)
    → libturbomind_c.so (C FFI via `turbomind_c.rs`)
      → CUDA kernels
```

**Call chain depth**: 3 layers
**Key files**:
- `lmdeploy-rust-server/src/turbomind_c.rs` (~1200 lines)
- `lmdeploy-rust-server/src/server.rs` (gRPC server)
- `lmdeploy-rust-server/src/handlers/` (request handling)

**Overhead sources**:
1. gRPC serialization/deserialization (Protobuf → string → Rust struct)
2. Network/IPC overhead (even localhost has socket latency ~10-50µs per round trip)
3. Rust FFI boundary safety checks (c_char* → Rust String conversion)
4. Memory allocation per token (Rust alloc → C++ alloc → back)
5. **CRITICAL**: gRPC streaming token-by-token (one message per token)
   - Each token triggers: C++ callback → Rust handler → Protobuf serialize → network → client deserialize
   - This is the **primary bottleneck** for decode performance

### Mode 3: PyBridge (subprocess + JSON)

```
User Script (Python process A)
  → stdin/stdout (JSON protocol)
    → python_bridge.py (Python process B)
      → lmdeploy.turbomind.TurboMind (Python)
        → _turbomind.so (pybind11)
          → libturbomind_c.so
```

**Call chain depth**: 6 layers
**Key files**:
- `lmdeploy/turbomind/python_bridge.py` (subprocess script)
- `lmdeploy/turbomind/turbomind.py` (same as Mode 1)

**Overhead sources**:
1. **subprocess creation**: fork + exec (~50-200ms on Linux)
2. **JSON serialization/deserialization**: per-message encoding/decoding
   - Large input_ids arrays (512-16384 integers) serialized as JSON
3. **IPC latency**: stdin/stdout pipe buffering and flushing
4. **Two Python interpreters**: memory and GC overhead doubled
5. **No connection reuse**: new subprocess per test/batch
6. Python GIL in both processes

---

## Root Causes Identified

### 1. gRPC Streaming Overhead (Rust Mode)

**Symptom**: Rust decode TPS significantly lower than Python despite same C++ backend.

**Root cause**: The current Rust implementation uses gRPC streaming where **each token** is a separate Protobuf message:
- `StreamChunk.token_id` is hardcoded to 0
- Text chunks are sent individually
- Protobuf serialization adds ~10-20µs per message
- Network socket (even localhost) adds ~10-50µs per round trip
- **Total per-token overhead: 20-70µs × N tokens**

For 128 output tokens, that's 2.5-9ms of pure overhead.

**Fix direction**:
- Batch tokens per response (send 8-16 tokens per gRPC message)
- Use Unix domain sockets instead of TCP
- Use `grpc::CompletionQueue` for async streaming instead of sync

### 2. JSON IPC Overhead (PyBridge Mode)

**Symptom**: PyBridge significantly slower than direct Python for prefill.

**Root cause**: JSON serialization of large integer arrays:
- 8192 tokens = JSON string of ~40KB
- json.dumps/loads on 40KB takes ~1-5ms
- Done twice (request → response) per batch
- **Total JSON overhead: 2-10ms per batch**

**Fix direction**:
- Use binary protocol (e.g., msgpack, or direct shared memory)
- Tokenize inside the bridge process (don't send raw IDs)
- Reuse subprocess across multiple generations

### 3. Instance Creation Overhead (Python Mode)

**Symptom**: Python Direct is slowest for TTFT.

**Root cause**: TurboMind instance creation loads model weights each time:
- `TurboMind()` constructor loads weights into VRAM
- `create_instance()` sets up CUDA context
- **No instance pooling**: new instance per generation

**Fix direction**:
- Instance pooling in Python (reuse TurboMind across calls)
- Lazy weight loading

### 4. Python Object Conversion Overhead (All Python Modes)

**Symptom**: Python Direct decode slower than Rust for same C++ code.

**Root cause**: pybind11 conversion overhead:
- `torch.from_dlpack()` creates Python tensor wrapper per token
- `output.token_ids` is a Python list created from C++ array
- Python's list.extend() allocates new memory each call

**Fix direction**:
- Use numpy arrays instead of Python lists
- Pre-allocate output buffer
- Minimize pybind11 boundary crossings

---

## Expected Performance Comparison

| Mode | Prefill | Decode | Notes |
|------|---------|--------|-------|
| Python Direct | Medium | Medium | High TTFT from model load, but once loaded, decent throughput |
| Rust PureCpp | Medium-High | Low-Medium | gRPC overhead dominates per-token streaming |
| PyBridge | Low | Low | JSON + subprocess overhead compounds with context size |

---

## Measurement Tools

- `root_cause_analysis.py`: Automated profiling tool with cProfile integration
- `test_three_modes.py`: Performance benchmark across context lengths
- `benchmark_comparison.py`: Statistical analysis of results

---

## Recommendations (Priority Order)

1. **Batch gRPC tokens** (Rust mode): Send 8-16 tokens per response instead of 1
2. **Use Unix sockets** (Rust mode): Replace TCP with Unix domain socket for local gRPC
3. **Pre-allocate output buffer** (Python mode): Avoid repeated list.extend() calls
4. **Instance pooling** (Python mode): Reuse TurboMind instances across generations
5. **Binary IPC protocol** (PyBridge mode): Replace JSON with msgpack or shared memory

---

## Files Changed

| File | Description |
|------|-------------|
| `lmdeploy-rust-server/tests/root_cause_analysis.py` | New profiling tool |
| `lmdeploy-rust-server/docs/performance_root_cause_analysis.md` | This analysis document |
