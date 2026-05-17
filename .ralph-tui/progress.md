# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **Config JSON Parsing Pattern**: Use string-based JSON parsing with `pattern.find()` + `substring()` approach. See `parse_json_int_field` and `parse_json_string_field` in `config/*.mbt` files. Pattern: find `"fieldname":`, skip whitespace, extract until `,` or `}`.
- **Model Config Structure**: `ModelConfig` in `src/config/model.mbt` is the canonical source of truth for model parameters. `ModelMetadata` in `src/model/model.mbt` derives from `ModelConfig`.
- **MoonBit Project Root**: `moon.mod.json` is at `lmdeploy-moonbit-server/` level, NOT at `lmdeploy-moonbit-server/lmdeploy_moonbit_server/`. Must `cd` to the directory containing `moon.mod.json` for `moon check/build`.
- **Tuple Returns for Complex Parsing**: When parsing multiple related config values, return them as a tuple (e.g., `(Int, Int, Int, Int, Int, Int, Int, Int)` for 8 model parameters) to avoid creating intermediate structs.
- **WASI FFI Stub Pattern**: Filesystem operations require WASI FFI. Use stub functions that return empty/defaults, with `use module::submodule` pattern for imports. See `config/config.mbt` `path_exists`/`read_file_content` stubs.
- **SafeTensors Format**: Last 8 bytes = header offset. JSON header contains `{"__metadata__": {...}, "tensor_name": {"dtype": "F32", "shape": [dims], "data_offsets": [begin, end]}}`. Tensor data follows header.
- **ModelWeight Hierarchy**: All weight types (LinearWeight, NormWeight, AttentionWeight, FfnWeight, MoeWeight, DeltaNetWeight, DecoderLayerWeight, ModelWeight) are defined in `src/model/model_weight.mbt`. Each type has config structs and builder functions.
- **AWQ Config JSON Structure**: `{"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemm", "symmetric": true, "zero_point": true, "pack": true}}` found in HuggingFace config.json
- **C++ Config Parsing Pattern**: Use `std::string::find()` + brace depth counting for nested JSON parsing. See `ReadAwqQuantConfig()` in `src/turbomind/capi/turbomind_c.cc`
- **LinearWeight Quantization Support**: `LinearWeight` already has `scales` and `zeros` parameters (for AWQ INT4 dequantization). Config via `DataFormat` and `MakeQuantDesc()`.
- **MoonBit Config Extension Pattern**: Add new field to struct → update default() → update new() → update from_json() → update from_env_overrides() → add accessors → update to_debug_string()
- **Rust Test Pattern for FFI**: Tests that require actual GPU/model should be marked with `#[ignore]`. Run with `cargo test --lib -- --ignored` to include them. Unit tests for types/values don't need GPU.

---

## 20260517 - lmdeploy-jog.1.6
- Implemented comprehensive tests for InitFromPath initialization sequence:
  - **Model state tests**: `test_init_from_path_state_transitions()` - verifies ModelState enum values and transitions
  - **Engine structure tests**: `test_engine_creation()`, `test_engine_reload_structure()` - validate engine creation/reload structure
  - **ModelInfo tests**: `test_model_info_structure()`, `test_model_info_default()` - test ModelInfo struct
  - **Init sequence documentation**: `test_init_sequence_documentation()` - documents expected C API initialization sequence
  - **FFI signature tests**: `test_process_weights_signature()`, `test_ffi_wrapper_availability()` - verify FFI wrapper types
- Added comprehensive FFI binding tests in turbomind_c.rs:
  - Error code enum value tests
  - Data type enum value tests
  - Memory type enum value tests
  - FFError formatting tests
  - TM_Error struct size tests
  - TM_SessionParam layout tests
  - ScheduleMetrics struct tests
  - InitFromPath sequence documentation test
  - C primitive type size tests
- **Files changed**:
  - `lmdeploy-rust-server/src/model/engine.rs` - Added 8 unit tests for initialization sequence
  - `lmdeploy-rust-server/src/turbomind_c.rs` - Added 15 unit tests for FFI bindings
- **Test Results**: 21 tests passing (6 in engine, 15 in turbomind_c)
- **Learnings**:
  - InitFromPath C++ implementation internally calls CreateContext, CreateRoot, ProcessWeights, and CreateEngine (lines 828-884 in turbomind_c.cc)
  - The Rust wrapper exposes these as separate methods but InitFromPath does the full sequence
  - Tests requiring actual GPU/model must be marked `#[ignore]` to prevent crashes in CI
  - `LD_LIBRARY_PATH` must include `/mnt/eaget-4tb/data/llm_server/lmdeploy/build/lib` for FFI tests to link
  - ModelWeight creation requires hidden_size and AWQ config from config.json
  - TM_Error struct has 256-byte message buffer + 4-byte code field (total 260 bytes)
---

## 20260517 - lmdeploy-jog.1.5
- Implemented AWQ 4-bit quantization parameter support across C++ and MoonBit:
  - **C++ TurboMind C API** (`src/turbomind/capi/turbomind_c.cc`):
    - `AwqQuantConfig` struct: bits, group_size, quant_method, version, symmetric, zero_point, pack
    - `ReadAwqQuantConfig()`: reads quantization_config block from config.json
    - `TM_TurboMind_InitFromPath()`: reads AWQ config, sets data_type appropriately
    - Comments added for AWQ dequantization during ProcessWeights step
  - **MoonBit Quantization Config** (`src/config/quantization.mbt`):
    - `AwqConfig`, `GptqConfig` structs with all quantization parameters
    - `QuantizationConfig` enum: None / Awq(AwqConfig) / Gptq(GptqConfig) / Other(String)
    - `parse_quantization_config()`, `parse_awq_config()`, `parse_gptq_config()` parsers
  - **MoonBit Model Config** (`src/config/model.mbt`):
    - Added `quantization` field to `ModelConfig` struct
    - Added `is_quantized()`, `quantization_method()` accessors
    - Added `parse_quantization_from_path()` integration
    - Updated `to_debug_string()` to show quantization info
- **Files changed:**
  - `src/turbomind/capi/turbomind_c.cc` - C++ AWQ config parsing and integration
  - `lmdeploy-moonbit-server/lmdeploy_moonbit_server/src/config/quantization.mbt` - NEW: Quantization config types and parsing
  - `lmdeploy-moonbit-server/lmdeploy_moonbit_server/src/config/model.mbt` - Quantization field integration
- **Learnings:**
  - AWQ 4-bit quantization: weights stored as INT4 packed, scales and zeros in FP16
  - `group_size: 128` means per-channel quantization with 128-weight groups
  - `zero_point: true` means AWQ uses asymmetric quantization with zeros tensor
  - `version: "gemm"` refers to the AWQ kernel version for dequantized GEMM

---

## 20260517 - lmdeploy-jog.1.4
- Created complete ModelWeight hierarchy structure for 40-layer decoder:
  - Configuration types: RopeConfig, AttentionConfig, FfnConfig, MoeConfig, DeltaNetConfig, NormConfig, LinearConfig, ModelWeightConfig
  - Weight types: NormWeight, LinearWeight, AttentionWeight, FfnWeight, MoeWeight, DeltaNetWeight, DecoderLayerWeight, ModelWeight
  - Builder functions: create_model_weight_config, create_attention_config, create_ffn_config, create_moe_config, create_deltanet_config, create_norm_config, create_linear_config
  - Validation: ModelWeight::prepare() and ModelWeight::verify() methods
- **Files changed:**
  - `src/model/model_weight.mbt` - NEW: Complete ModelWeight hierarchy (~1034 lines)
  - `src/model/moon.pkg.json` - Added import for weights module
- **Learnings:**
  - TurboMind C++ ModelWeight hierarchy mirrors the model structure: ModelWeight (root) -> DecoderLayerWeight (per layer) -> sub-components (attention, ffn, moe, delta_net, norms)
  - Qwen3.6-35B has 40 transformer layers, default num_layers in ModelWeight::new()
  - AttentionWeight supports both standard MHA and MLA (Multi-Head Latent Attention) via kv_lora_rank check
  - DeltaNet (linear attention) is used in hybrid models like Qwen3.6-35B (Gated Delta Net with SSM layers)
  - MoeWeight contains experts array for MoE models, each expert is an FfnWeight
---

## 20260517 - lmdeploy-jog.1.3
- Implemented safetensors weight loading framework for GPU:
  - Created `src/model/weights.mbt` with complete safetensors parsing infrastructure
  - `WeightDtype` enum: Float32, Float16, BFloat16, Int8, UInt8, Int32, Int64 with size_bytes()
  - `TensorShape` struct: dimensions array with rank() and num_elements()
  - `DataOffsets` struct: begin/end offsets for tensor data in file
  - `WeightTensor` struct: individual tensor metadata with loading state tracking
  - `SafeTensorsHeader` struct: parsed header with tensor array and metadata map
  - `SafeTensorsFile` struct: file representation with path and header
  - `WeightContainer` struct: manages all safetensors files for a model, tensor index lookup
  - `parse_safetensors_header()`: parses safetensors JSON header into structured metadata
  - `parse_tensor_metadata()`, `parse_shape_array()`, `parse_data_offsets()`: field parsers
  - `find_matching_brace()`: nested brace matching for JSON parsing
  - `load_safetensors_files()`: discover and load all safetensors files in model directory
  - `load_weights_to_gpu()`: GPU memory loading (stubs, needs TurboMind FFI)
- **Files changed:**
  - `src/model/weights.mbt` - NEW: Complete safetensors weight loading module (~450 lines)
  - `src/model/engine.mbt` - Added `load_weights()` and `get_weights()` methods using weights module
- **Learnings:**
  - SafeTensors format stores header offset in last 8 bytes of file (little-endian Int64)
  - Tensor `data_offsets` are relative to header start, not file start
  - `__metadata__` key is optional and can be present in safetensors headers
  - MoonBit Int64 type requires separate parsing from Int (different literal syntax: `0L`)
  - WASI filesystem FFI needed for actual file I/O; current implementation uses stubs
---
- Extended `ModelConfig` struct to include all parameters needed for `ModelWeightConfig`:
  - Added `num_kv_heads` (for GQA/MQA support)
  - Added `intermediate_size` (MLP hidden dimension)
  - Added `bos_token_id`, `eos_token_id` (special token IDs)
  - Added `sliding_window` (window size for sliding window attention)
  - Added `tie_word_embeddings` (whether input and output embeddings are tied)
  - Added `use_mrope` (multi-rope support)
  - Added `tp_size`, `tp_rank` (tensor parallelism configuration)
  - Added `data_type` (model data type, e.g., "float16")
- **Files changed:**
  - `src/config/model.mbt` - Extended struct with 10 new fields, added parsing functions for new config.json fields, added accessor methods, added `parse_json_bool_field` helper
  - `src/model/engine.mbt` - Updated init() to reference new config fields (commented out as placeholders)
- **Learnings:**
  - HuggingFace config.json uses `num_key_value_heads` for GQA/MQA models
  - `intermediate_size` is the hidden dimension of the MLP layer, typically 4x `hidden_size`
  - Token IDs (`bos_token_id`, `eos_token_id`) are needed for proper text generation
  - Tensor parallelism (`tp_size`, `tp_rank`) is needed for multi-GPU inference
  - Sliding window attention uses -1 to indicate disabled (no limit)
  - MoonBit tuples can hold up to 8 elements; extended return type from 4 to 8 for model config parsing

---

## 20260517 - lmdeploy-jog.1.1
- Implemented config.json parsing for full model configuration (num_layers, num_heads, vocab_size, hidden_size, head_dim)
- **Files changed:**
  - `src/config/model.mbt` - Added 5 new fields to ModelConfig struct, parsing functions for config.json, accessor methods, updated from_json and from_env_overrides
  - `src/model/model.mbt` - Updated ModelMetadata::from_config to use cfg.vocab_size() instead of hardcoded value
- **Learnings:**
  - HuggingFace config.json uses `num_hidden_layers` and `num_attention_heads` as standard field names
  - Some models use nested `text_config` blocks (e.g., Qwen2 multimodal), requiring fallback parsing
  - MoonBit build must run from `lmdeploy-moonbit-server/` directory (where `moon.mod.json` lives)
  - WASI filesystem FFI is needed for actual file reading; current implementation uses stub returning empty string with fallback to defaults
---