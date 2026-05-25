# CUDA Graph 集成建议 - 待完成

## 当前状态
- TASK-4 (CUDA Graph) 已标记为 in_progress
- 功能未完整实现
- 是提升 prefill 性能的关键优化之一

## 建议的实现路径

### 1. C++ 层 CUDA Graph 支持
在 `src/turbomind/kernels/` 或新文件中添加：
- CUDA Graph capture/launch wrapper
- 为不同 input size 缓存 graph
- 最小化重新分配

### 2. FFI 暴露
在 `src/turbomind/capi/turbomind_c.{h,cc}` 中添加：
```c++
// turbomind_c.h
struct TM_CudaGraph;
TM_CudaGraph* TM_CudaGraph_Create(TM_ModelRequest* req);
int TM_CudaGraph_Capture(TM_CudaGraph* graph, TM_TensorMap* input_tensors);
int TM_CudaGraph_Launch(TM_CudaGraph* graph, TM_TensorMap* input_tensors);
void TM_CudaGraph_Destroy(TM_CudaGraph* graph);
```

### 3. Rust 端集成
在 `lmdeploy-rust-server/src/turbomind_c.rs` 添加 FFI bind，
在 `cpp_engine.rs` 添加缓存逻辑，

### 4. 性能预期
- 预期: kernel 启动延迟降低 50%+
- GPU 利用率提升 20%+
- 这可以减少 Python TurboMind 与 Rust 的性能差距

## 其他已完成的优化
- ✓ TASK-1: DLPack zero-copy tensor 传递
- ✓ TASK-2: pre-tokenization + pre-allocation
- ✓ TASK-3: RMSNorm 分析完成 (无冗余)
- ✓ TASK-5: KV Cache direct mapping (已实现)
- ✓ TASK-FIX: CUDA 库依赖修复
- ✓ TASK-BENCH: 性能基线建立