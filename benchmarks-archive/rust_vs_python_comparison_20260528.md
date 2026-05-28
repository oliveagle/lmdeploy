# Rust vs Python Prefill Performance Comparison

**Date**: 2026-05-28  
**Model**: Qwen3.6-35B-A3B-AWQ  
**GPU**: Tesla PG503-216 (32GB)

## Analysis Summary

### Rust Benchmark Status

The Rust `prefill_benchmark` binary **cannot run** due to C++ TurboMind engine crash during model loading.

**Root Cause**: Tensor name mismatch between safetensors file format and C++ engine expectations.

**Details**:
- Safetensors format: `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`
- C++ engine expects: `layers.0.attention.w_qkv.weight` OR `layers.0.linear_attn.in_proj_qkv.weight`

**Error Log**:
```
[C-API] ERROR: Cannot find w_qkv.weight param for layers.7.attention
[C-API] ERROR: Cannot find w_qkv.weight param for layers.3.attention
...
[TM][FATAL][buffer.h:70] 'data_' Must be non NULL
```

The C++ engine has `layer_is_linear_attn` configuration that should handle DeltaNet linear attention layers, but the weight loading code appears to have a bug where some layers are being searched for standard attention weights instead of linear attention weights.

### Python Baseline (from Archive)

From `benchmarks-archive/python-turbomind_35b_awq_20260526/`:

| Input Length | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|--------------|-----------|-----------------|----------------|
| 512          | 79.9      | 6,408           | 42.6           |
| 1024         | 139.2     | 7,356           | 42.2           |
| 4096         | 402.0     | 10,190          | 40.5           |
| 8192         | 649.4     | 12,614          | 39.8           |

**Calculation**:
- Prefill (tok/s) = input_len / (ttft_ms / 1000)
- Decode (tok/s) = 1000 / tpot_ms

### Existing Rust Benchmark Results

From previous successful runs (when C++ engine worked):

| Context | TTFT (ms) | Prefill (tok/s) |
|---------|-----------|-----------------|
| 1K      | 84.8      | 12,075          |
| 4K      | 189.3     | 21,636          |
| 8K      | 336.1     | 24,376          |

**Note**: These results appear to be from a working configuration, possibly with different model weights or C++ engine version.

## Recommendations

1. **Fix C++ Engine**: Debug the `layer_is_linear_attn` configuration and weight loading path
2. **Use Python Baseline**: The Python TurboMind baseline is reliable and consistent
3. **Rust Optimization**: Once C++ engine is fixed, Rust wrapper should match or exceed Python performance

## Conclusion

**Rust prefill_benchmark cannot run** due to C++ engine bug. Comparison deferred until weight loading is fixed.

The Python TurboMind baseline provides consistent 6K-12K tok/s prefill performance across context lengths.
