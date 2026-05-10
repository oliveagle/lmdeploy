# EP=4 Testing Guide

This guide explains how to test Expert Parallelism (EP=4) support for Qwen3.6-35B-A3B-AWQ in LMDeploy.

## Quick Start

### 1. Configuration Verification (No GPU Required)

Verify that all EP configuration components are correctly implemented:

```bash
python tests/test_lmdeploy/test_turbomind/verify_ep_config.py
```

Expected output:
```
✅ messages.py verified
✅ config.py verified
✅ converter.py verified
✅ module.py verified
✅ turbomind.py verified
✅ turbomind.cc verified
✅ llama_params.h verified
🎉 All EP=4 configuration checks passed!
```

### 2. Unit Tests (No GPU Required)

Run unit tests for EP configuration:

```bash
python tests/test_lmdeploy/test_turbomind/test_ep_moe.py
```

Or with pytest:
```bash
pytest tests/test_lmdeploy/test_turbomind/test_ep_moe.py -v
```

### 3. Model Loading Test (GPU Required)

Test model loading and basic inference:

```bash
export QWEN36_A3B_AWQ_PATH=/path/to/Qwen3.6-35B-A3B-AWQ
python tests/test_lmdeploy/test_turbomind/test_ep4_model_loading.py
```

### 4. Performance Benchmark (GPU Required)

Compare EP=4 vs TP=4 performance:

```bash
python tests/test_lmdeploy/test_turbomind/benchmark_ep4.py \
    --model-path $QWEN36_A3B_AWQ_PATH \
    --output ep4_benchmark_results.json
```

### 5. Generate Test Report

Generate a comprehensive test report:

```bash
python tests/test_lmdeploy/test_turbomind/ep4_test_report.py
```

## Test Scripts Overview

| Script | Purpose | GPU Required |
|--------|---------|--------------|
| `verify_ep_config.py` | Verify EP configuration chain | No |
| `test_ep_moe.py` | Unit tests for EP configuration | No |
| `test_ep4_model_loading.py` | Model loading and quality tests | Yes |
| `benchmark_ep4.py` | Performance benchmark suite | Yes |
| `ep4_test_report.py` | Generate test report | No |

## EP Configuration Example

```python
from lmdeploy import TurbomindEngineConfig, pipeline

# Create engine config with EP=4
engine_config = TurbomindEngineConfig(
    ep=4,              # Expert Parallelism size
    tp=1,              # Tensor Parallelism size
    device_num=4,      # Total number of GPUs
    session_len=2048,  # Session length
    max_batch_size=1,  # Maximum batch size
    quant_policy=8,    # KV cache quantization (K=4bit, V=2bit)
)

# Create pipeline
pipe = pipeline(
    model_path='/path/to/model',
    backend='turbomind',
    engine_config=engine_config,
)

# Run inference
response = pipe(['Hello, how are you?'])
print(response[0])
```

## Expected Results

### Memory Usage

| Configuration | Memory per GPU | Total Memory |
|---------------|----------------|--------------|
| EP=4, TP=1 | ~15 GB | ~60 GB |
| TP=4, EP=1 | ~18 GB | ~72 GB |

### Expert Distribution (256 experts, EP=4)

| Rank | Expert Range | Count |
|------|--------------|-------|
| 0 | [0, 63] | 64 |
| 1 | [64, 127] | 64 |
| 2 | [128, 191] | 64 |
| 3 | [192, 255] | 64 |

### Output Quality

Expected:
- ✅ No garbled text (e.g., all '!' characters)
- ✅ Diverse vocabulary
- ✅ Semantically coherent responses

Known Issue (PyTorch backend):
- ❌ PyTorch EP=4 produces garbled output (all '!')
- ✅ Turbomind EP=4 should NOT have this issue

## Troubleshooting

### Model Loading Fails

1. Check GPU memory: `nvidia-smi`
2. Verify 4x GPUs are available
3. Check model path is correct
4. Ensure Turbomind C++ extension is compiled

### Garbled Output

1. Verify backend is 'turbomind' (not 'pytorch')
2. Check EP configuration is correct
3. Verify expert range calculation
4. Compare with TP=4 baseline

### Out of Memory

1. Reduce `session_len`
2. Reduce `max_batch_size`
3. Enable `quant_policy=8` (KV cache quantization)
4. Use larger GPUs (A100 40GB instead of V100 16GB)

## Performance Comparison

Run benchmark to compare EP=4 vs TP=4:

```bash
python tests/test_lmdeploy/test_turbomind/benchmark_ep4.py \
    --model-path $QWEN36_A3B_AWQ_PATH \
    --output ep4_vs_tp4.json
```

Expected results:
- EP=4 should have similar or better throughput than TP=4
- EP=4 should use less memory per GPU
- EP=4 may have slightly higher latency due to all_reduce

## Next Steps

After EP=4 validation passes:
1. Document EP configuration in user guide
2. Add EP examples to documentation
3. Test with other MoE models (Qwen3.5-27B-AWQ)
4. Optimize EP all_reduce communication
5. Test multi-node EP configuration

## References

- Design Doc: `.ralph-tui/STORY-002_DESIGN.md`
- Progress Log: `.ralph-tui/progress.md`
- Test Report: `tests/test_lmdeploy/test_turbomind/EP4_TEST_REPORT.txt`
