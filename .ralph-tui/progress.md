# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### Python Bridge JSON Protocol

When implementing Python subprocess communication via stdin/stdout:
1. **Disable lmdeploy logging to stdout** - it corrupts JSON parsing:
   ```python
   import logging
   _logger = logging.getLogger('lmdeploy')
   for handler in _logger.handlers:
       if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
           handler.stream = sys.stderr
   ```
2. **Disable prefix caching for linear attention models** - Qwen3.6 crashes with it enabled
3. **Set all parallel config explicitly** - dp=1, cp=1 required even for TP=1 to avoid assertions
4. **Use list not numpy array for input_ids** - `async_stream_infer` expects Sequence, not ndarray

### Benchmark Timing Measurement

For accurate TTFT (Time To First Token) measurement:
- **Use streaming API** - non-streaming APIs only return total elapsed time
- **TTFT estimation from total time is inaccurate** - subprocess overhead (~1.5s) distorts results
- **Decode speed calculation**: `(output_tokens * 1000) / (total_time_ms - prefill_time_ms)`

### Module::create() Pattern for Module Creation

When creating child modules, use `turbomind::core::Module::create(cfg)` to create modules through the registry:
```cpp
turbomind::core::ModuleListConfig layers_cfg;
auto layers_list_unique = turbomind::core::Module::create(layers_cfg);
auto* layers_list = static_cast<turbomind::core::ModuleList*>(layers_list_unique.get());
```
Then add via `parent->add_child("name", std::move(module))`.

### Tensor Data Access Pattern

Use `tensor.raw_data()` instead of `tensor.data<T>()` when the dtype is unknown at compile time:
```cpp
auto tensor = param.get();
if (tensor && tensor.raw_data()) {
    std::memcpy(tensor.raw_data(), data, copy_size);
}
```

### DataFormat Initialization

`DataFormat` is a struct, not an enum. Initialize with `turbomind::DataFormat{}` for default (plain) format:
```cpp
output_cfg.format = turbomind::DataFormat{};  // Not kPlain
```

### std::min Type Deduction

When using `std::min` with mixed types, explicitly cast to the common type:
```cpp
size_t copy_size = std::min(data_size, static_cast<size_t>(tensor.byte_size()));
```

---

### LMDeploy Model Loading Architecture (C++ vs Python)

**Python TurboMind API** (`lmdeploy/turbomind/turbomind.py`):
- `_from_hf()` → `get_tm_config()` → `ModelLoader` → `model_loader.export()` → `_process_weights()` → `_create_engine()`
- The `ModelLoader` reads safetensors, maps HF weight names to TurboMind names, and exports tensors
- `empty_init=True` skips weight loading; use `update_params()` to push weights later

**C++ C API** (`src/turbomind/capi/turbomind_c.cc`):
- `TM_TurboMind_Create()` → `TM_TurboMind_InitFromPath()`: CreateContext → CreateRoot → ProcessWeights → CreateEngine
- **GAP**: `InitFromPath` only creates a `ModelWeight` module with config metadata but never loads actual weight data from files
- C API has `SafetensorsReader` (lines 580-714) but it's NOT integrated into `InitFromPath`
- `TM_TurboMind_InitFromHF` (line 817) uses Python bridge but returns NOT_IMPLEMENTED after execution

**Model conversion** (`convert_hf_to_turbomind.py`):
- Calls Python `TurboMind(model_path, ...)` which triggers `_from_hf` path with full weight loading
- Converted workspace contains `.bin` files that `InitFromPath` can load
- Rust server code calls `InitFromPath` directly on the safetensors path, bypassing conversion

### Weight Serialization Pattern

**Python weight_serializer.py**:
- Reads HF safetensors files and converts to TurboMind .bin format
- Creates three outputs:
  1. `config.yaml` - Model configuration (hidden_size, num_layers, etc.)
  2. `*.bin` files - Binary weight files (one per safetensors shard)
  3. `weight_index.json` - Index mapping weight names to file locations
- Handles BF16/FP16/FP32 dtypes with proper numpy conversions

**C++ weight_serializer.h/cc**:
- Provides `SerializeWeightsToBin()` and `LoadWeightsFromBin()` functions
- Uses `core::Module` for weight tree management
- Placeholder for full C++ serialization (Python version is more complete)

### AWQ Quantization Handling

- AWQ 4-bit weights use INT4 storage with scales/zeros
- Python `AWQFormat` in `weight_format.py` handles normalization
- C++ `AwqQuantConfig` struct parses config.json for quantization parameters

---

## 2026-05-18 - lmdeploy-7n4
- Analyzed AWQ model loading failure root cause in Rust Server
- Files examined: `turbomind_c.cc`, `engine.rs`, `turbomind.py`, `convert_hf_to_turbomind.py`
- **Root cause**: `TM_TurboMind_InitFromPath` in C API does not load weight data from safetensors - it only creates empty `ModelWeight` structure without tensor data
- **Three fix options**:
  1. Integrate `SafetensorsReader` into `InitFromPath` to load weights directly (C++)
  2. Convert HF models via Python `TurboMind` API first, then load from converted workspace
  3. Complete `TM_TurboMind_InitFromHF` Python bridge (currently returns NOT_IMPLEMENTED)
- Option 2 recommended since Python `ModelLoader` already handles AWQ weight mapping correctly
---

## 2026-05-18 - lmdeploy-k7d
- Implemented Python bridge for TurboMind model loading to fix AWQ weight loading issue
- Files created/modified:
  - `lmdeploy/turbomind/python_bridge.py` - New Python subprocess bridge wrapping TurboMind API
  - `lmdeploy-rust-server/src/model/python_bridge.rs` - New Rust module for subprocess communication
  - `lmdeploy-rust-server/src/model/engine.rs` - Replaced C API engine with Python bridge
- **Root cause**: C API's `TM_TurboMind_InitFromPath` creates empty ModelWeight without loading weight data
- **Solution**: Use Python TurboMind API via subprocess bridge (stdin/stdout JSON protocol)
- **Rust compilation**: Verified with `cargo check` - 0 errors, 45 warnings (style warnings only)

## 2026-05-18 - lmdeploy-jpv
- Implemented TurboMind weight serialization to .bin format
- Files created:
  - `src/turbomind/utils/weight_serializer.h` - Header with serialization API
  - `src/turbomind/utils/weight_serializer.cc` - Implementation (placeholder)
  - `lmdeploy/turbomind/weight_serializer.py` - Full Python implementation
  - `src/turbomind/capi/turbomind_c.cc` - Added `TM_ExportWeightsToBin()` C API function
  - `src/turbomind/capi/turbomind_c.h` - Added function declaration
- **Learnings:**
  - Python `weight_serializer.py` successfully converts HF safetensors to .bin format
  - BF16 tensors need special handling: `tensor.float().numpy().view(np.uint16).tobytes()`
  - C++ namespace `core::Module` forward declaration requires `namespace core { class Module; }` syntax
  - C API integration requires linking `weight_serializer` library in `capi/CMakeLists.txt`
- Verification: Tested on Qwen3.6-35B-A3B-AWQ model, generated 9 .bin files (32.5GB total) + config.yaml + weight_index.json

---

## 2026-05-18 - lmdeploy-5wx
- Analyzed Python TurboMind ModelWeight construction and C API gaps
- Files examined: `model_weight.h/cc`, `decoder_layer_weight.h`, `model_loader.py`, `turbomind_c.cc`
- **Learnings:**
  - ModelWeight tree structure: tok_embeddings, norm, output, layers[40] (AttentionWeight, FfnWeight, DeltaNetWeight)
  - C API's `InitFromPath` creates empty ModelWeight with only config metadata, no tensor data
  - C API has SafetensorsReader (lines 580-714) but it's NOT integrated into InitFromPath
  - Python path: ModelLoader.export() → create_checkpoint() → model.model(Prefix) → C++ weight binding
  - Documentation: `.records/model_weight_structure_20260518.md`
---

## 2026-05-18 - lmdeploy-zmu
- Implemented C++ weight loading in `TM_TurboMind_InitFromPath()`
- Files changed: `src/turbomind/capi/turbomind_c.cc` - Complete rewrite of InitFromPath to build full module tree and load weights
- **Implementation:**
  - Module tree construction: Creates ModelWeight → layers[48] → {attention, feed_forward, attention_norm, ffn_norm}
  - Safetensors loading: Uses existing SafetensorsReader to read tensor data
  - Weight name mapping: Maps HF names (model.layers.0.self_attn.q_proj.weight) to TM paths (layers.0.attention.q_proj.weight)
  - Parameter allocation: Allocates tensor params via Param::alloc() and copies weight data
- **Learnings:**
  - Module::create(cfg) uses registry to create modules through typed config
  - tensor.raw_data() works when dtype unknown; tensor.data<T>() requires compile-time type
  - DataFormat is a struct, not enum - initialize with DataFormat{} for plain format
  - std::min needs explicit type casting when mixing size_t with ssize_t
- Verification: Build succeeds, 65 tests pass
---

## 2026-05-18 - lmdeploy-207
- Verified InitFromPath implementation completeness
- **Status**: Implementation complete (from lmdeploy-zmu, lmdeploy-bor)
- Files verified: `turbomind_c.cc` (lines 1226-1434), `model_weight.cc` (prepare method)
- **Verified components**:
  - Module tree construction: ModelWeight → layers[N] → {attention, feed_forward, attention_norm, ffn_norm}
  - Safetensors loading: SafetensorsReader class with JSON header parsing
  - Weight name mapping: MapHuggingFaceWeightToTurboMind() function
  - AWQ config: AwqQuantConfig struct for quantization parameters
- **Weight mapping verified**:
  - `model.layers.0.self_attn.q_proj.weight` → `layers.0.attention.q_proj.weight`
  - `model.layers.0.mlp.gate_proj.weight` → `layers.0.feed_forward.w1.weight`
  - `model.layers.0.mlp.up_proj.weight` → `layers.0.feed_forward.w3.weight`
  - `model.layers.0.mlp.down_proj.weight` → `layers.0.feed_forward.w2.weight`
- **Learnings**:
  - InitFromPath flow: CreateContext → CreateRoot → build tree + load weights → ProcessWeights → CreateEngine
  - AttentionWeight children: w_qkv, wo, q_proj, k_proj, v_proj, q_a_proj, q_b_proj, kv_a_proj, q_norm, k_norm
  - FfnWeight children: w1, w3, w2, w1w3
  - Build status: Compiles successfully, turbomind_c.so built

---

## 2026-05-18 - lmdeploy-2tt
- Implemented Rust Server benchmark via Python Bridge for Qwen3.6-35B-A3B-AWQ
- Files created:
  - `BENCHMARK_RUST_BRIDGE_20260518.json` - Benchmark results in Python-compatible format
- Files modified:
  - `lmdeploy-rust-server/src/model/benchmark.rs` - Added metrics API, fixed summary grouping
  - `lmdeploy-rust-server/src/model/engine.rs` - Added `generate_with_metrics()` and `tokenizer()` methods
  - `lmdeploy-rust-server/src/model/python_bridge.rs` - Added `generate_with_metrics()` method
  - `lmdeploy/turbomind/python_bridge.py` - Fixed logging, prefix caching, parallel config
- **Benchmark Results** (3 runs per scenario, averaged):
  | Context | TTFT (ms) | Prefill (tps) | Decode (tps) | Total Time (ms) |
  |---------|-----------|--------------|--------------|-----------------|
  | 1K      | 1478.48   | 249.2        | 59.4         | 12320.68        |
  | 4K      | 1547.19   | 943.9        | 56.7         | 12893.26        |
  | 8K      | 1656.81   | 1760.2       | 53.0         | 13806.77        |
- **Comparison with Python TurboMind**:
  - Python TTFT: 70-191 ms vs Rust Bridge: 1478-1657 ms (TTFT estimation issue)
  - Python Decode: 40.6-41.2 tps vs Rust Bridge: 53.0-59.4 tps (Rust ~30% faster?)
  - The TTFT difference is due to estimation method - Rust bridge reports total time including overhead
- **Learnings:**
  - Python bridge subprocess communication adds ~1.5s overhead per request
  - TTFT measurement requires streaming API to be accurate (not implemented in bridge)
  - Prefix caching must be disabled for Qwen3.6 (linear attention model)
  - TurbomindEngineConfig requires explicit dp/cp settings for TP=1 to avoid assertion failures
  - lmdeploy logging to stdout interferes with JSON protocol - must redirect to stderr
  - Benchmark summary grouping needs tolerance for actual vs target token counts

---
