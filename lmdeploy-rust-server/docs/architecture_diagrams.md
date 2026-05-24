# Architecture Diagrams

## Mode 1: Python Direct

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  USER SCRIPT (Python)                                                       │
│  - Application logic                                                       │
│  - Prompt construction                                                     │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ Python API call
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  lmdeploy.turbomind.TurboMind (Python)                                     │
│  - turbomind.py wrapper (~800 lines)                                       │
│  - TurboMind instance creation                                             │
│  - stream_infer() method                                                   │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ pybind11 FFI
                             │ Python objects → C++ types
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  _turbomind.so (pybind11 / CPython module)                                 │
│  - pybind11 bindings                                                       │
│  - List<Tensor> → std::vector<Tensor> conversion                          │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ C function call
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  libturbomind_c.so (C++ engine)                                            │
│  - turbomind_c.cc (C API)                                                  │
│  - TurboMind C++ class                                                     │
│  - Attention / MLP kernels                                                 │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ CUDA launch
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  CUDA Kernels (GPU)                                                         │
│  - FlashAttention                                                          │
│  - GEMM / gemm_lite                                                        │
│  - MoE routing (if applicable)                                             │
└─────────────────────────────────────────────────────────────────────────────┘

OVERHEAD SOURCES:
  ⚠️  Python object allocation/deallocation per token
  ⚠️  pybind11 conversion: Python list → C++ vector
  ⚠️  TurboMind instance created per generation (no pooling)
  ⚠️  List.extend() reallocates memory each call
```

---

## Mode 2: Rust + C++ (PureCpp)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  gRPC CLIENT (Python/any)                                                  │
│  - Generates request protobuf                                              │
│  - Receives stream responses                                               │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ gRPC network call
                             │ TCP (even localhost!)
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  lmdeploy-rust-server (Rust)                                               │
│  - server.rs: gRPC service                                                 │
│  - turbomind_c.rs: FFI bindings (~1200 lines)                              │
│  - handlers/: request processing                                           │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ extern "C" FFI call
                             │ c_char* → Rust String
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  libturbomind_c.so (C++ engine)                                            │
│  - Same as Python Direct                                                   │
│  - turbomind_c.cc (C API)                                                  │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ CUDA launch
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  CUDA Kernels (GPU)                                                         │
│  - Same as Python Direct                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

OVERHEAD SOURCES:
  ⚠️  gRPC serialization: Protobuf encode/decode per message
  ⚠️  TCP socket latency: ~10-50µs per round trip (even localhost!)
  ⚠️  ⚠️  CRITICAL: One gRPC message PER TOKEN
       For 128 tokens: 128 messages × (10-50µs) = 1.3-6.4ms overhead
  ⚠️  Rust FFI safety checks (string conversion)
  ⚠️  Memory allocation: Rust alloc → C++ alloc per token
```

---

## Mode 3: PyBridge (subprocess + JSON)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  USER SCRIPT (Process A - Python)                                          │
│  - Application logic                                                       │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ JSON write to stdin
                             │ subprocess.Popen(..., stdin=PIPE)
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  PIPE (stdin/stdout)                                                        │
│  - Buffered I/O                                                            │
│  - JSON serialization/deserialization                                      │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ fork + exec (first time only)
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  python_bridge.py (Process B - Python)                                     │
│  - Reads JSON from stdin                                                   │
│  - Parses command                                                          │
│  - Forwards to lmdeploy API                                               │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ Python API call
                             │ (Same as Mode 1)
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  lmdeploy.turbomind.TurboMind (Python)                                     │
│  - Same as Mode 1                                                          │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ (Same chain as Mode 1)
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│  ... _turbomind.so → libturbomind_c.so → CUDA ...                          │
└─────────────────────────────────────────────────────────────────────────────┘

OVERHEAD SOURCES:
  ⚠️  subprocess fork+exec: ~50-200ms startup cost
  ⚠️  JSON serialization: Large input_ids arrays
       8192 tokens → ~40KB JSON string → 1-5ms serialization
  ⚠️  Two Python interpreters: Double memory and GC overhead
  ⚠️  Pipe buffering: Additional latency on read/write
  ⚠️  No subprocess reuse: New process per generation
  ⚠️  JSON decode: Response parsing on client side
```

---

## Performance Impact Summary

| Mode | Call Depth | Main Overhead | Prefill Impact | Decode Impact |
|------|-----------|---------------|----------------|---------------|
| Python Direct | 4 layers | pybind11 conversion | Medium (list→vector) | Medium (per-token) |
| Rust+C++ | 3 layers | gRPC per-token | Low | **HIGH** (1 msg/token) |
| PyBridge | 6 layers | JSON + subprocess | **HIGH** (40KB JSON) | **HIGH** (IPC per token) |

---

## Key Insights

1. **Rust mode's decode bottleneck is NOT the C++ or Rust code**
   - The C++ engine is identical across all modes
   - The bottleneck is gRPC streaming: one message per token
   - Fix: Batch tokens per response (send 8-16 tokens per message)

2. **PyBridge is slowest overall due to compounding overheads**
   - subprocess startup + JSON + extra Python layer
   - For prefill: JSON dominates (40KB for 8K context)
   - For decode: IPC dominates (one round trip per token)

3. **Python Direct is surprisingly competitive**
   - Once model is loaded, pybind11 overhead is ~10-20µs per token
   - No network/IPC latency
   - Instance pooling would make it fastest for TTFT

---

## Fixes (Priority Order)

1. **[Rust] Batch gRPC tokens** - Change from 1 msg/token to 8-16 tokens/msg
   - Impact: Decode TPS +300-500%
   - Effort: Medium (modify StreamChunk structure)

2. **[Rust] Use Unix domain sockets** - Replace TCP with UDS
   - Impact: Latency -50%
   - Effort: Low (change bind address)

3. **[Python] Instance pooling** - Reuse TurboMind instances
   - Impact: TTFT -80%
   - Effort: Medium (requires state management)

4. **[PyBridge] Binary protocol** - Replace JSON with msgpack
   - Impact: Overall +20-30%
   - Effort: Low-Medium

5. **[All] Pre-allocate output buffers** - Avoid repeated allocations
   - Impact: +5-10%
   - Effort: Low
