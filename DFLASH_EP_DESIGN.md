# Turbomind EP=4 Support Design

> **Date**: 2026-05-10
> **Status**: Design Phase
> **Goal**: Add Expert Parallelism (EP) support to Turbomind backend for MoE models

## Background

### Current State
- **Turbomind**: Only supports TP (Tensor Parallelism) for MoE weights
- **PyTorch backend**: Supports EP but has output quality issues (all "!!!")
- **User need**: EP=4, TP=1 configuration for 4x V100 16GB GPUs
- **Model**: Qwen3.6-35B-A3B-AWQ (256 experts × 40 layers)

### Why EP for MoE

**Problem**: With TP=4 only:
- Each GPU stores ALL 256 experts
- Expert weights sharded only by `inter_size / tp_size`
- 4x V100 16GB is insufficient (~18 GB/GPU needed)

**Solution**: EP=4:
- Each GPU stores only 64 experts (256 / 4)
- No TP sharding for MoE (tp_size=1 for MoE)
- ~5 GB/GPU for MoE weights instead of ~18 GB/GPU

## Architecture Design

### 1. Expert Partitioning Strategy

**Formula**: Same as PyTorch backend
```cpp
expert_per_rank = (num_experts + ep_size - 1) / ep_size
first_expert = ep_rank * expert_per_rank
last_expert = min(first_expert + expert_per_rank, num_experts)
```

**Example**: 256 experts, ep_size=4
- Rank 0: experts [0, 1, ..., 63]
- Rank 1: experts [64, 65, ..., 127]
- Rank 2: experts [128, 129, ..., 191]
- Rank 3: experts [192, 193, ..., 255]

### 2. Forward Pass Flow

```
Input [batch, hidden]
    ↓
Router (global, all ranks compute same routing)
    ↓
Filter to local experts only
    ↓
MoE GEMM (local experts only)
    ↓
AllReduce across EP group
    ↓
Output [batch, hidden]
```

### 3. Communication Pattern

**No All-to-All needed** - simpler than full EP implementation:
- Router runs globally (all ranks compute same routing)
- Each rank processes only its assigned experts
- `all_reduce` combines results at the end

**Why this works**:
- Router weights are small (hidden_size × num_experts)
- `all_reduce` is cheaper than all-to-all for MoE
- Matches PyTorch backend pattern

## Implementation Plan

### Phase 1: C++ Core Changes

#### 1.1 Add EP Parameters

**File**: `src/turbomind/models/llama/llama_params.h`

```cpp
struct EngineParam {
    // ... existing params ...

    // Expert Parallelism
    int mlp_ep_size = 1;
    int mlp_ep_rank = 0;
};

struct MoeParam {
    // ... existing params ...

    // EP support
    int ep_size = 1;
    int ep_rank = 0;
};
```

#### 1.2 Modify LlamaWeight

**File**: `src/turbomind/models/llama/LlamaWeight.h/.cc`

```cpp
// Add members
int mlp_ep_size_ = 1;
int mlp_ep_rank_ = 0;

// Constructor extracts from EngineParam
mlp_ep_size_ = engine_param.mlp_ep_size;
mlp_ep_rank_ = engine_param.mlp_ep_rank;
```

#### 1.3 Modify LlamaDecoderLayerWeight

**File**: `src/turbomind/models/llama/LlamaDecoderLayerWeight.h/.cc`

```cpp
// Add members
int mlp_ep_size_;
int mlp_ep_rank_;

// Constructor extracts from EngineParam
mlp_ep_size_ = engine.mlp_ep_size;
mlp_ep_rank_ = engine.mlp_ep_rank;
```

#### 1.4 Modify MoeFfnWeight

**File**: `src/turbomind/models/llama/LlamaDenseWeight.h/.cc`

```cpp
struct MoeFfnWeight {
    // ... existing ...

    // EP: Expert range for this rank
    int ep_first_expert_ = 0;
    int ep_num_experts_ = 0;

    // EP: AllReduce across EP group
    void EpAllReduce();
};

// Constructor: compute expert range
MoeFfnWeight::MoeFfnWeight(..., int tp_size, int tp_rank, ...) {
    // ... existing ...

    if (tp_size > 1 && /* EP mode */) {
        // In EP mode, tp_size is actually ep_size
        int expert_per_rank = (total_experts + tp_size - 1) / tp_size;
        ep_first_expert_ = tp_rank * expert_per_rank;
        ep_num_experts_ = std::min(expert_per_rank, total_experts - ep_first_expert_);
    }
}
```

#### 1.5 Modify MoeFfnLayer

**File**: `src/turbomind/models/llama/moe_ffn_layer.h/.cc`

```cpp
class MoeFfnLayer {
    // ... existing ...

    // EP: AllReduce across EP group
    void EpAllReduce(Tensor& output);

    // EP: Filter experts to local only
    void FilterToLocalExperts(...);
};

void MoeFfnLayer::Forward(ForwardParam& p) {
    // ... router runs globally ...

    if (ep_size_ > 1) {
        // Filter to local experts
        FilterToLocalExperts(...);
    }

    // ... MoE GEMM with local experts only ...

    if (ep_size_ > 1) {
        // AllReduce across EP group
        EpAllReduce(p.output);
    }
}
```

### Phase 2: Python Configuration

#### 2.1 TurbomindEngineConfig

**File**: `lmdeploy/messages.py`

```python
@dataclass
class TurbomindEngineConfig:
    # ... existing ...
    ep_size: int = 1  # Expert parallelism size
```

#### 2.2 ModelConfig

**File**: `lmdeploy/turbomind/deploy/config.py`

```python
@dataclass
class ModelConfig:
    # ... existing ...
    mlp_ep_size: int = 1  # MoE expert parallelism
    mlp_ep_rank: int = 0
```

#### 2.3 Parameter Conversion

**File**: `lmdeploy/turbomind/deploy/parameter.py`

```python
def create_workspace_config(...):
    # ... existing ...
    config['mlp_ep_size'] = model_config.mlp_ep_size
    config['mlp_ep_rank'] = model_config.mlp_ep_rank
```

### Phase 3: Weight Loading

#### 3.1 Module Export

**File**: `lmdeploy/turbomind/deploy/module.py`

```python
class MoeFfn:
    def apply(self, ...):
        # ... existing ...

        if ep_size > 1:
            # Only export this rank's experts
            expert_per_rank = (num_experts + ep_size - 1) // ep_size
            first_expert = ep_rank * expert_per_rank
            last_expert = min(first_expert + expert_per_rank, num_experts)

            for i in range(first_expert, last_expert):
                # Export expert i only
                ...
        else:
            # Export all experts (existing code)
            ...
```

#### 3.2 Source Model Loading

**File**: `lmdeploy/turbomind/deploy/source_model/qwen.py`

```python
class Qwen3_5MoeModel:
    def _from_huggingface(self, ...):
        # ... existing ...

        if ep_size > 1:
            # Only load this rank's experts
            expert_per_rank = (num_experts + ep_size - 1) // ep_size
            first_expert = ep_rank * expert_per_rank
            # ... load only experts [first_expert, first_expert + expert_per_rank)
        else:
            # Load all experts (existing code)
            ...
```

## Communication Implementation

### EP AllReduce

Since Turbomind doesn't use NCCL directly like PyTorch, we need to:

**Option 1**: Reuse existing `AllReduce` infrastructure
- Turbomind already has TP AllReduce
- Create EP process group similar to TP group

**Option 2**: Use CUDA NCCL directly
- Simpler for prototype
- Matches PyTorch backend pattern

**Recommended**: Start with Option 2 for prototype, move to Option 1 for production

## Key Challenges

### 1. Router Execution

**Question**: Should router run globally or per-rank?

**Answer**: Globally
- Router weights are small (~256 × 4096 × 2 bytes = 2MB)
- All ranks compute same routing decisions
- Each rank filters to local experts only

### 2. Expert ID Remapping

**Global to Local mapping**:
```cpp
int global_expert_id = ...;  // from router
int local_expert_id = global_expert_id - ep_first_expert_;
```

For experts not on this rank, skip computation.

### 3. Combine/AllReduce

**Current**: `Combine()` scales by `1.f / tp_size_`

**EP mode**: Scale by `1.f / ep_size_` instead

```cpp
void MoeFfnLayer::Combine(ForwardParam& p) {
    float scale = 1.f / (ep_size_ > 1 ? ep_size_ : tp_size_);
    // ...
}
```

## Testing Strategy

### 1. Unit Tests

- Test expert partitioning formula
- Test weight loading with EP
- Test forward pass output correctness

### 2. Integration Tests

- Load Qwen3.6-35B-AWQ with EP=4, TP=1
- Verify memory usage < 16GB per GPU
- Verify output quality (not all "!!!")

### 3. Performance Tests

- Benchmark EP=4 vs TP=4
- Measure AllReduce overhead
- Compare with PyTorch EP

## Success Criteria

1. ✅ Model loads successfully with EP=4
2. ✅ Memory usage < 16GB per GPU on 4x V100
3. ✅ Output quality is correct (not all "!!!")
4. ✅ Performance comparable to TP=4
5. ✅ Compatible with existing TP-only code

## Files to Modify

### C++ Headers
- `src/turbomind/models/llama/llama_params.h`
- `src/turbomind/models/llama/LlamaWeight.h`
- `src/turbomind/models/llama/LlamaDecoderLayerWeight.h`
- `src/turbomind/models/llama/LlamaDenseWeight.h`
- `src/turbomind/models/llama/moe_ffn_layer.h`

### C++ Sources
- `src/turbomind/models/llama/LlamaWeight.cc`
- `src/turbomind/models/llama/LlamaDecoderLayerWeight.cc`
- `src/turbomind/models/llama/LlamaDenseWeight.cc`
- `src/turbomind/models/llama/moe_ffn_layer.cc`

### Python
- `lmdeploy/messages.py`
- `lmdeploy/turbomind/deploy/config.py`
- `lmdeploy/turbomind/deploy/parameter.py`
- `lmdeploy/turbomind/deploy/module.py`
- `lmdeploy/turbomind/deploy/source_model/qwen.py`
- `lmdeploy/turbomind/turbomind.py`

## References

- PyTorch EP: `lmdeploy/pytorch/distributed.py`
- PyTorch MoE EP: `lmdeploy/pytorch/backends/cuda/moe/default.py`
- Turbomind MoE: `src/turbomind/models/llama/moe_ffn_layer.cc`
