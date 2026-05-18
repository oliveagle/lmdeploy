# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### ModelWeight Module Tree Structure

The complete ModelWeight module tree is built by Python TurboMind Builders:

```
ModelWeight
├── tok_embeddings (param) - [vocab_size, hidden_size]
├── output (LinearWeight) - LM Head
│   └── weight, bias, scales, zeros (params)
├── norm (NormWeight) - Final RMS norm
│   └── weight (param)
└── layers (ModuleList)
    └── 0..N (DecoderLayerWeight)
        ├── attention_norm (NormWeight)
        │   └── weight (param)
        ├── ffn_norm (NormWeight)
        │   └── weight (param)
        ├── attention (AttentionWeight)
        │   ├── q_norm, k_norm, q_a_layernorm, kv_a_layernorm (NormWeight)
        │   └── w_qkv, wo, q_proj, q_a_proj, q_b_proj, kv_a_proj (LinearWeight)
        │       └── weight, bias, scales, zeros (params)
        ├── feed_forward (FfnWeight)
        │   └── w1, w3, w2, w1w3 (LinearWeight)
        │       └── weight, bias, scales, zeros (params)
        ├── linear_attn (DeltaNetWeight) - optional for Qwen3.5-MoE
        │   ├── norm (NormWeight)
        │   ├── in_proj_all, out_proj (LinearWeight)
        │   └── conv1d, A_log, dt_bias (params)
        └── moe_ffn (MoeWeight) - optional for MoE models
            ├── gate (LinearWeight)
            ├── experts (ModuleList of FfnWeight)
            ├── shared_experts (FfnWeight)
            └── routing (FfnWeight)
```

**Key Finding**: C API `InitFromPath()` only creates empty shell modules. The Python Builders (TextModelBuilder, AttentionBuilder, FfnBuilder, MoeBuilder, NormBuilder) are responsible for:
1. Creating child modules via `_tm.create_module(cfg)`
2. Committing weight data via `_copy_shard_to_param()`
3. Attaching children via `add_child_raw()`

---

## 2026-05-18 - lmdeploy-5wx
- **Implemented**: Analysis of Python TurboMind weight loading and ModelWeight module tree
- **Files changed**: `.ralph-tui/progress.md` - Documented complete ModelWeight module tree structure
- **Learnings**:
  - Python TurboMind uses Builder pattern (TextModelBuilder, AttentionBuilder, FfnBuilder, MoeBuilder, NormBuilder)
  - Builders create child modules via `_tm.create_module(cfg)` and commit weight data via `_copy_shard_to_param()`
  - Children are attached via `add_child_raw()` after weight data is committed
  - C API `InitFromPath()` only creates empty shell modules without actual weight data
  - The missing piece in C API is that it doesn't build the LinearWeight children (w_qkv, wo, w1, w2, w3) with their params
  - AttentionWeight has 9 LinearWeight children (w_qkv, wo, q_proj, q_a_proj, q_b_proj, kv_a_proj) + 5 NormWeight children (q_norm, k_norm, q_a_layernorm, kv_a_layernorm)
  - FfnWeight has 4 LinearWeight children (w1, w3, w2, w1w3)
  - DeltaNetWeight has 2 LinearWeight children (in_proj_all, out_proj) + 1 NormWeight + 3 params (conv1d, A_log, dt_bias)

---

## 2026-05-19 - lmdeploy-qp8
- **Implemented**: Rust Server SSE Streaming 支持
- **Files changed**:
  - `lmdeploy/turbomind/python_bridge.py` - Added `generate_stream()` async generator and command handler
  - `lmdeploy-rust-server/src/model/python_bridge.rs` - Added `GenerateStream` command, `BridgeStreamChunk` type, `generate_stream()` method
  - `lmdeploy-rust-server/src/model/engine.rs` - Implemented `generate_stream()` to use Python bridge streaming
- **Learnings**:
  - Python TurboMind's `async_stream_infer()` with `stream_output=True` yields tokens one-by-one
  - Need to use `async for` with `yield` in Python async generators (not `return` with values)
  - Rust streaming uses `mpsc::channel` + `ReceiverStream` to bridge sync thread reading to async Stream
  - The HTTP handler was already wired up; just needed to implement the underlying `generate_stream()` method
  - SSE streaming sends `data: {json}` lines with `event: chat.completion.chunk` prefix
  - TTFT (Time To First Token) is tracked by recording time when first token is received
  - Use `filter_map` to skip empty tokens in the stream before sending to clients

---

## 2026-05-18 - lmdeploy-jpv
- **Implemented**: Fixed weight_serializer.py for nested config handling (Qwen3.5 MoE)
- **Files changed**:
  - `lmdeploy/turbomind/weight_serializer.py` - Added nested config support (text_config, model_config)
  - `tests/test_weight_serializer.py` - Created tests for serialization
- **Learnings**:
  - Qwen3.5 MoE models have nested config structure - actual model params are in `text_config` sub-object
  - Weight serializer must handle both flat configs (standard HF) and nested configs (multimodal/MoE models)
  - Safetensors I/O uses 8-byte header size + JSON header + binary data format
  - Tensor alignment to 64-byte boundaries is important for GPU loading performance
  - The Python weight_serializer.py is fully functional for HF→TM .bin conversion
  - C++ weight_serializer.cc is still a placeholder - uses Python bridge for actual conversion

---

