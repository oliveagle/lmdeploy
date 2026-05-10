# Expert Parallelism (EP)

Expert Parallelism (EP) is a parallelization strategy for Mixture of Experts (MoE) models that distributes different experts across different GPUs. This is different from Tensor Parallelism (TP), which shards individual tensors across GPUs.

## Overview

In MoE models, each layer contains multiple expert networks. EP allows you to:

- **Reduce memory per GPU**: Each GPU only stores a subset of experts
- **Scale to larger models**: Run models with more experts than would fit in a single GPU
- **Maintain throughput**: Expert routing allows efficient GPU utilization

### EP vs TP

| Feature | Expert Parallelism (EP) | Tensor Parallelism (TP) |
|---------|------------------------|------------------------|
| **Sharding granularity** | Expert-wise | Tensor-wise |
| **Communication pattern** | All-to-All (expert routing) | All-Reduce (every layer) |
| **Best for** | MoE models with many experts | Dense models or MoE with few experts |
| **Memory efficiency** | High (each GPU stores 1/EP of experts) | Medium (each GPU stores all experts) |
| **Latency impact** | Low (only during expert routing) | Higher (all-reduce every layer) |

## Configuration

### Basic Usage

```python
from lmdeploy import TurbomindEngineConfig, pipeline

# Create engine config with EP=4
engine_config = TurbomindEngineConfig(
    ep=4,              # Expert Parallelism size
    tp=1,              # Tensor Parallelism size
    device_num=4,      # Total number of GPUs (ep * tp)
    session_len=2048,  # Session length
    max_batch_size=1,  # Maximum batch size
    quant_policy=8,    # KV cache quantization (4+8 for MoE models)
)

# Create pipeline
pipe = pipeline(
    model_path='/path/to/qwen3.6-35b-a3b-awq',
    backend='turbomind',
    engine_config=engine_config,
)

# Run inference
response = pipe(['Hello, how are you?'])
print(response[0])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ep` | int | 1 | Number of GPUs for expert parallelism. Must divide `device_num` evenly. |
| `ep_rank` | int | 0 (auto) | Rank of current GPU in EP group (0 to ep-1). Auto-calculated from device ID. |
| `tp` | int | 1 | Number of GPUs for tensor parallelism. Must divide `device_num` evenly. |
| `device_num` | int | 1 | Total number of GPUs. Must equal `ep * tp`. |

### Expert Distribution

For a MoE model with 256 experts and EP=4:

| GPU (Rank) | Expert Range | Number of Experts |
|------------|--------------|-------------------|
| 0 | [0, 63] | 64 |
| 1 | [64, 127] | 64 |
| 2 | [128, 191] | 64 |
| 3 | [192, 255] | 64 |

The expert range is calculated as:
```python
experts_per_rank = (total_experts + ep_size - 1) // ep_size
ep_first_expert = ep_rank * experts_per_rank
ep_num_experts = min(experts_per_rank, total_experts - ep_first_expert)
```

## Performance Characteristics

### Memory Usage

Memory usage per GPU for Qwen3.6-35B-A3B-AWQ (256 experts):

| Configuration | Memory per GPU | Total Memory |
|---------------|----------------|--------------|
| EP=4, TP=1 | ~15 GB | ~60 GB |
| TP=4, EP=1 | ~18 GB | ~72 GB |

**Key insight**: EP reduces memory per GPU because each GPU stores only 1/EP of the experts.

### Throughput and Latency

| Metric | EP=4, TP=1 | TP=4, EP=1 |
|--------|-----------|-----------|
| **Throughput** | Similar or better | Baseline |
| **Latency** | Slightly higher (all_reduce) | Lower (no all_reduce) |
| **Batch capacity** | Higher (less memory) | Lower (more memory) |

**Trade-offs**:
- EP has slightly higher latency due to all_reduce communication during expert routing
- EP allows larger batch sizes due to reduced memory footprint
- For MoE models, EP typically provides better overall throughput than TP

## Supported Models

EP is supported for Turbomind backend MoE models including:

- Qwen3.5-27B-AWQ
- Qwen3.6-35B-A3B-AWQ
- Other Qwen MoE variants

## Hardware Requirements

### Minimum Configuration

- **4x V100 16GB** (EP=4, TP=1)
  - Model: Qwen3.6-35B-A3B-AWQ
  - Session length: 2048
  - Batch size: 1
  - KV quantization: 4+8 bit

### Recommended Configuration

- **4x A100 40GB** (EP=4, TP=1)
  - Supports longer sessions (4096+)
  - Larger batch sizes
  - Production-ready performance

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. **Reduce session length**: `session_len=2048` → `session_len=1024`
2. **Enable KV quantization**: `quant_policy=8` (4+8 bit)
3. **Reduce batch size**: `max_batch_size=1` → `max_batch_size=1` (already minimal)
4. **Use larger GPUs**: A100 40GB instead of V100 16GB

### Garbled Output

If you see garbled text (e.g., all '!' characters):

1. **Verify backend**: Use `backend='turbomind'` (not 'pytorch')
2. **Check EP configuration**: Ensure `ep` and `device_num` are correct
3. **Verify model**: Ensure you're using a supported MoE model

**Known issue**: PyTorch backend with EP=4 produces garbled output. Use Turbomind backend instead.

### Configuration Errors

Common configuration errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `device_num != ep * tp` | Invalid parallelism configuration | Ensure `device_num = ep * tp` |
| `ep > device_num` | EP size exceeds available GPUs | Reduce `ep` or increase `device_num` |
| `device_num` doesn't match GPUs | Mismatch between config and hardware | Set `device_num` to actual GPU count |

## Advanced Topics

### EP + TP Combinations

Currently, LMDeploy supports:
- **EP=4, TP=1**: Pure expert parallelism (recommended for MoE models)
- **EP=1, TP=4**: Pure tensor parallelism (baseline)

**Note**: EP + TP combinations (e.g., EP=2, TP=2) are not yet fully validated for Turbomind backend.

### Multi-Node EP

Multi-node EP configuration is not yet tested. Single-node EP is fully supported.

### KV Cache Quantization

For MoE models, use `quant_policy=8` (4-bit K, 8-bit V) to reduce memory:

```python
engine_config = TurbomindEngineConfig(
    ep=4,
    tp=1,
    quant_policy=8,  # 4+8 bit KV quantization
)
```

## References

- [TurboMind Configuration](./turbomind_config.md)
- [KV Quantization](../quantization/kv_quant.md)
- [MoE Model Support](../supported_models/supported_models.md)
