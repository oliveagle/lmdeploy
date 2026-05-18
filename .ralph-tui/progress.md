# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

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

