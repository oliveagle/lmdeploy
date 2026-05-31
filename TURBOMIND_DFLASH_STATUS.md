# LMDeploy + DFlash Turbomind 集成状态

**日期**: 2026-05-10
**状态**: ✅ Turbomind 编译成功，DFlash C++ 集成完成

## 已完成的工作

### 1. DFlash C++ 源文件 ✅
- `src/turbomind/models/llama/DFlashDraftModel.{h,cu}` - Draft model 实现
- `src/turbomind/models/llama/DFlashDraftWeight.{h,cc}` - 权重结构
- `src/turbomind/models/llama/dflash_kernels.{h,cu}` - CUDA kernels
- `src/turbomind/models/llama/unified_decoder_dflash.{h,cc}` - Decoder 扩展

### 2. Turbomind 集成 ✅
- `LlamaWeight.{h,cc}` - 已添加 DFlashDraftWeight 支持
- `LlamaDecoder.cc` - 已实现 speculative 解码框架
- `unified_decoder.{h,cc}` - 已集成 DFlash 逻辑
- `CMakeLists.txt` - 已添加 DFlash 源文件到编译列表

### 3. 编译成功 ✅
```bash
✓ Turbomind 编译成功
✓ import lmdeploy.lib._turbomind 成功
✓ import lmdeploy.turbomind.TurboMind 成功
```

## 测试结果

### ❌ Turbomind + AWQ (TP=4 on V100)
**状态**: 失败 - 内存不足

**错误**:
```
CUDA runtime error: out of memory
```

**原因**:
- Qwen3.6-35B-A3B-AWQ 需要 ~18 GB/GPU (TP=4)
- V100 只有 16GB 显存

## 待完成工作

### 1. Python 集成
- [ ] 添加 `SpeculativeConfig` 到 `lmdeploy/messages.py`
- [ ] 修改 `turbomind.py` Python 接口支持 DFlash
- [ ] 添加 DFlash draft model 权重加载逻辑

### 2. 测试
- [ ] 使用更大显存的 GPU (A100 40GB+)
- [ ] 或使用更多 GPU (TP=8)
- [ ] 或测试更小的模型

### 3. 性能优化
- [ ] 优化 DFlash CUDA kernels
- [ ] 调整 aux layers 位置
- [ ] 优化 draft token 数量

## 建议

**硬件需求**:
- A100 40GB 或更大显存的 GPU
- 或使用 8x V100 (TP=8)

**下一步**:
1. 完成 Python 集成
2. 在合适的硬件上测试
3. 性能基准测试

## 相关文件

- 测试脚本: `test_turbomind_dflash.py`
- DFlash 源文件: `src/turbomind/models/llama/DFlash*`
- C++ 集成: `src/turbomind/models/llama/unified_decoder.cc`
