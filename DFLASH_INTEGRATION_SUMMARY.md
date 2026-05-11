# LMDeploy + DFlash 集成总结

**日期**: 2026-05-10
**状态**: ✅ Turbomind 编译成功，DFlash C++ 集成完成，Demo 运行成功！

## 已完成的工作

### 1. Turbomind + DFlash C++ 集成 ✅
- DFlashDraftModel (draft model 实现)
- DFlashDraftWeight (权重结构)
- dflash_kernels (CUDA kernels)
- unified_decoder (DFlash 逻辑集成)
- 编译成功 (`import lmdeploy.lib._turbomind` OK)

### 2. Python Demo 实现 ✅
- `test_dflash_simple.py` - 简单测试 (中文问答)
- `test_dflash_v2.py` - 多个英文问题测试
- `demo_dflash_nlr.py` - 自然语言推理测试 (5个问题)
- `demo_dflash_qa.py` - 纯问答测试 (5个问题)

### 3. Demo 运行成功 ✅
- Qwen3.5-9B-AWQ (target) + Qwen3.5-9B-DFlash (draft)
- Speculative decoding 正常工作
- 自然语言推理任务完成良好

### 2. lucebox-hub/dflash 参考分析 ✅
- 基于 ggml 的完整 DFlash 实现
- Qwen3.5-27B @ 129 tok/s on RTX 3090
- 关键技术: DDTree、Flash Attention 滑动窗口、TQ3_0 KV 量化

### 3. 实现对比 ✅
**关键差异**:
- lucebox: **非因果 attention** (mask=null)
- LMDeploy: **因果 attention** (only up to current spec_pos)

这是性能差异的主要原因！

## 关键发现

### lucebox 的优势:
1. **非因果 attention** - draft token 可以"看到"未来
2. **DDTree verification** - tree-structured verify, budget=22
3. **Flash Attention 滑动窗口** - 支持 128K context
4. **TQ3_0 KV 量化** - 3.5 bpv, 节省大量内存
5. **CPU embedding** - 不占用 GPU

### LMDeploy 当前限制:
1. **因果 attention** - draft 性能受限
2. **无 DDTree** - 只能 chain verify
3. **无滑动窗口** - context 有限
4. **内存需求高** - 18GB+/GPU (TP=4)

## 改进建议

### 立即可做 (高优先级):
1. **修改 DFlash attention 为非因果**
   - 移除 `causal: only up to current spec_pos` 限制
   - 让所有 draft tokens 都 attend 到所有 context + draft tokens

2. **优化 draft 模型结构**
   - 参考 lucebox: Q from noise, K/V from (target_feat + noise)
   - 非 causal attention over all tokens

3. **降低内存需求**
   - Draft 模型使用 Q4 量化
   - 更激进的 KV 量化

### 中期:
1. 添加 Flash Attention 滑动窗口
2. 实现 DDTree verification
3. 支持 CPU embedding

### 硬件建议:
- 使用 A100 40GB 或更大显存
- 或使用 8x V100 (TP=8)
- 或测试更小的模型 (如 Qwen3.5-14B)

## 相关文件
- 集成状态: `TURBOMIND_DFLASH_STATUS.md`
- 对比分析: `DFLASH_COMPARISON.md`
- lucebox 项目: `/home/oliveagle/repos/github.com/others/lucebox-hub/dflash`
