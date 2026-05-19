## 2026-05-20 - lmdeploy-7pt
- **Investigation**: Compared Python bridge vs C API weight loading from safetensors
- **Python Bridge Path** (`turbomind.py` → `model_loader.py` → `checkpoint.py`):
  1. `create_checkpoint()` picks `SafetensorsCheckpoint` based on file patterns
  2. `SafetensorsCheckpoint.__init__` opens all shards via `safe_open(shard, 'pt')`, builds flat `dict[str, torch.Tensor]` (mmap-backed)
  3. `ModelLoader.export()` calls `self.model.model(Prefix(ckpt))` - traverses Python model tree
  4. Each module reads tensors via `Prefix.get(key)` → `ckpt.get(key).cuda()` - PyTorch handles GPU transfer
  5. Per-model `_loader_mappings` (regex functions) transform HF names before storage
  6. `ProcessWeights()` → `CreateEngine()` via ThreadPoolExecutor over GPU count
- **C API Path** (`turbomind_c.cc:InitFromPath`):
  1. Manually constructs entire module tree (ModelWeight, layers, attention, ffn) with hardcoded shapes
  2. Attaches ModelWeight to ModelRoot as "text_model" child
  3. Creates ContextGuard, allocates tok_embeddings param
  4. `LoadWeightsFromSafetensors()` uses C safetensors.h API
  5. `MapHuggingFaceWeightToTurboMind()` does string replacements for key mapping
  6. Finds params via `for_each_param()`, copies data via `cudaMemcpy`/`std::memcpy`
- **Key Differences**:
  - Python uses PyTorch's `.cuda()` for GPU transfer; C API uses manual `cudaMemcpy`
  - Python has per-model `_loader_mappings` for flexible key mapping; C API has a single hardcoded function
  - Python builders framework automatically constructs correct module tree; C API hardcodes everything
  - Python supports 40+ model families via module registry; C API currently only Llama-like
- **Files examined**:
  - `lmdeploy/turbomind/turbomind.py` - Python entry point, weight loading orchestration
  - `lmdeploy/turbomind/model_loader.py` - ModelLoader class with export() method
  - `lmdeploy/turbomind/checkpoint.py` - SafetensorsCheckpoint/PytorchCheckpoint implementations
  - `src/turbomind/capi/turbomind_c.cc` - InitFromPath and LoadWeightsFromSafetensors
  - `src/turbomind/utils/weight_serializer.py` - Standalone utility (not used by either path)
- **Learnings**:
  - `SafetensorsCheckpoint` uses mmap-backed dict - no host-RAM copy happens up front
  - `Prefix.get(key)` appends `.cuda()` automatically - all tensor reads go to GPU
  - Python `_loader_mappings` are per-model regex functions stored in model builder classes
  - C API `MapHuggingFaceWeightToTurboMind` is a single function with many `string::find`/`replace` calls
  - `weight_serializer.py` is unused - it would convert HF safetensors to TM .bin format but neither path uses it
  - The segfault was from `std::memcpy` on GPU tensors (fixed in lmdeploy-6xm), not from architectural differences
---
