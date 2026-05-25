# CUDA Graph 集成状态 - 2026-05-25

## 当前状态

### FFI 层（已完成）
- ✅ `turbomind_c.h` - C API 声明已定义
- ✅ `turbomind_c.rs` - Rust FFI 绑定已实现
  - `TM_CudaGraph_Capture`
  - `TM_CudaGraph_Launch` 
  - `TM_CudaGraph_Destroy`
  - `CudaGraphHandle` 类型

### C++ 引擎层（未实现）
- ❌ `turbomind_c.cc:2984` - 返回错误 "CUDA Graph capture not yet implemented"
- ❌ `turbomind_c.cc:3016` - 返回错误 "CUDA Graph launch not yet implemented"

## 为什么需要引擎层支持

CUDA Graph 捕获需要：
1. **记录整个计算图** - 捕获 kernel 序列和依赖关系
2. **内存管理** - 确保所有内存分配在 capture 期间固定
3. **动态输入处理** - 不同 input size 需要不同的 graph

这需要修改 TurboMind 引擎内部（`src/turbomind/`），添加：
- Graph capture 逻辑
- Graph cache（按 input size 索引）
- Memory pool 固定

## 性能预期
- Kernel 启动延迟降低 50%+
- GPU 利用率提升 20%+

## 实现优先级
这是一个**大型重构任务**，需要：
1. 理解 TurboMind 引擎架构
2. 添加 CUDA Graph 支持
3. 验证正确性和性能

建议作为独立的引擎层任务处理。
