# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **TurboMind C API InitFromPath sequence**: `CreateContext → CreateRoot → BuildModelWeight → ProcessWeights → CreateEngine` — all 5 steps executed in order. Each step depends on the previous. C++ side handles everything (CUDA context, GPU weight loading, engine creation).
- **AWQ quantization policy**: `quant_policy=4` for AWQ 4-bit. `AwqConfig` struct in C++ (`bits=4`, `group_size=128`, `version="gemm"`, `symmetric=true`, `zero_point=true`, `pack=true`) must match HF config.json `quantization_config` fields.
- **Rust FFI wrappers**: RAII pattern with `struct Wrapper(*mut TM_Type)` + `Drop` impl. Use `FFResult<T> = Result<T, FFError>` for error handling. All unsafe FFI calls wrapped in safe Rust API.
- **Model weight hierarchy**: 40 decoder layers with attention (qkv_proj, o_proj), ffn (gate_proj, up_proj, down_proj), and linear_attn MoE (gate_up_proj, down_proj). Each layer has corresponding weight tensors in safetensors.

---

## 2026-05-17 - lmdeploy-jog
- **Epic closed**: LMDeploy Rust Server - TurboMind C API Integration
- **User story closed**: lmdeploy-jog.1 - C++ 模型权重加载 - 修复 TM_TurboMind_InitFromPath()
- **Files changed**:
  - `src/turbomind/capi/turbomind_c.h` - C API header with FFI types and functions
  - `src/turbomind/capi/turbomind_c.cc` - C++ implementation with InitFromPath, ProcessWeights, CreateEngine, AWQ support
  - `lmdeploy-rust-server/src/turbomind_c.rs` - Rust FFI bindings with RAII wrappers
  - `lmdeploy-rust-server/src/model/engine.rs` - TurboMindEngine with full C API integration
  - `lmdeploy-rust-server/src/lib.rs` - Module organization (error, tokenizer, server, model)
  - `lmdeploy-rust-server/src/api/server.rs` - HTTP server with OpenAI-compatible endpoints
- **Implementation summary**:
  - C API (`libturbomind_c.so`) provides direct FFI access to TurboMind engine
  - Rust FFI bindings (`turbomind_c.rs`) wrap all C types with RAII pattern
  - Engine implementation (`engine.rs`) handles model loading, config parsing, inference
  - AWQ 4-bit quantization support via `quant_policy` and `AwqConfig`
  - OpenAI-compatible HTTP server with `/chat/completions` and `/completions` endpoints
- **Learnings:**
  - Pattern discovered: TurboMind InitFromPath executes full 5-step init sequence internally
  - Pattern discovered: RAII wrappers in Rust for FFI safety (EngineConfig, TurboMind, TensorMap, GenConfig, ModelRequest)
  - Gotcha: AWQ quant config fields in config.json must match C++ AwqConfig defaults exactly
  - Gotcha: TurboMind-converted models don't include tokenizer files — must load from HF model path
  - Gotcha: build.rs links against libturbomind_c.so + CUDA libs (cudart, cublasLt, cuda)

---

