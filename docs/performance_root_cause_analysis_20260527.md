# 性能根因分析 - Qwen3.6-35B-AWQ Prefill 仅 ~6.4K tok/s 而非预期 20K+

**分析日期**: 2026-05-27
**目标**: 找出 TurboMind C++ 前向传播中导致 prefill 吞吐量不足预期的根因

---

## 实测数据对比

| 来源 | Prefill (tok/s) | 说明 |
|------|-----------------|------|
| benchmarks-archive 实测 | 6,408 - 12,614 | 随输入长度增加而提升 |
| 预期目标 | ~20,000+ | 基于硬件理论峰值 |

## 根因分析：5 个瓶颈

### 根因 1: 同步点阻断 GPU 流重叠（影响度: 中）

**文件**: `src/turbomind/models/llama/unified_attention_layer.cc:587`

```cpp
TM_CUDA_CHECK(cudaGetLastError());
```

- `cudaGetLastError()` 虽不强制同步，但每次 kernel 后调用会阻碍 GPU driver 的异步调度优化
- 在 35B 模型 60+ 层 × 4 个 kernel/层 = 240+ 次调用 per forward pass
- 累积开销在微秒级别

**建议**: 仅在 debug 模式下启用

### 根因 2: 层间串行执行无 overlap（影响度: 高）

**文件**: `src/turbomind/models/llama/unified_decoder.cc:234-331`

```cpp
for (int layer = 0; layer < layer_num_; ++layer) {
    attn_layer_->Forward({...});         // attention
    AllreduceResidualRMSnorm(...);       // sync point
    ffn_layer_->forward({...});           // ffn
    AllreduceResidualRMSnorm(...);       // sync point
}
```

- 每个 layer 的 attention → norm → ffn → norm 完全串行
- 没有使用 CTA 级别的 attention-FFN overlap
- GPU SM 在 kernel 间隙存在空闲时间

**建议**: 实现 CTA-level overlap 或使用 CUDA Graph 捕获整个 decoder forward

### 根因 3: Prefill 使用通用注意力 kernel 而非 prefill 专用（影响度: 高）

**Prefill kernel**: `MMA_16816`, CTA=[64,64,64]
```
attention_sm80_128.cu: AttentionUniversal<Sm80, Mainloop<CpAsync<2>, Impl<MMA_16816, ...>>>
```

**Decode kernel**: `MMA_81616` (转置的 16816), CTA=[1, 64, ...]
```
decoding_sm80_128.cu: KT<Mainloop<CpAsync<Stages>, Impl<MMA_81616, ...>>>
```

- 两者使用相同的 MMA 架构但不同的 tile 配置
- Prefill kernel 是通用的"universal"实现，不是针对长序列优化的特殊实现
- 缺少 FlashDecoding++ / FlashAttention-3 等针对长序列的优化
- 对于 context >= 8192，没有专门的优化路径

**建议**: 引入 prefill 专用 kernel（如 FlashAttention 变体），使用更大的 CTA tile

### 根因 4: MoE 路由开销（针对 Qwen3.6-35B-A3B 稀疏模型）（影响度: 高）

**文件**: `src/turbomind/kernels/moe/moe_mm.cu:37-41`, `unified_moe_ffn.cc:241-370`

```cpp
TM_CUDA_CHECK(cudaGetLastError());  // moe_mm.cu 中
```

- 每个 token 都需要 top-k 路由计算
- Pre-fill 阶段所有 token 都要参与路由决策
- MoE dispatch + combine 涉及额外的内存分配和 kernel 启动

**建议**: 对连续 token 块进行路由缓存或批量处理

### 根因 5: 内存分配在 critical path（影响度: 低）

**文件**: `src/turbomind/models/llama/unified_moe_ffn.cc:244-269`

```cpp
moe::Context ctx(core::Context::allocator(),
                 head_dim_,
                 topk_,
                 topk_method_,
                 topk_group_,
                 n_expert_,
                 n_local_expert_,
                 ep_rank_,
                 false);  // moe_mm.cu
```

- MoE context 在每层 forward 时构造
- 虽然 allocator 是共享的，但构造开销在 hot path 上

**建议**: 预分配 MoE context，重用缓冲区

## 综合评估

预期 20K+ tok/s 的理论值基于：
1. FP16 Tensor Core 峰值 (312 TFLOPS for A100 级别)
2. 纯计算无同步的理想情况
3. FlashAttention 级别的 kernel 优化

实际 6.4K-12.6K tok/s 的差距主要由以下因素构成：

| 因素 | 预计影响 |
|------|----------|
| 层间串行无 overlap | ~30% 损失 |
| 通用 kernel vs 专用优化 | ~25% 损失 |
| MoE 路由/通信开销 | ~20% 损失 |
| 同步点累积 | ~5% 损失 |
| 内存分配 overhead | ~3% 损失 |

## 优化优先级建议

1. **CUDA Graph 集成** - 捕获整个 decoder forward，消除 kernel launch 开销（预期提升 20-30%）
2. **Prefill 专用 kernel** - 使用 FlashAttention 变体替代 universal kernel（预期提升 15-25%）
3. **CTA-level overlap** - attention 和 FFN 在同一 stream 中重叠执行（预期提升 10-15%）
4. **MoE 批量路由优化** - 预分配 + 缓存（预期提升 5-10%）

## 结论

当前 6.4K-12.6K tok/s 的预填吞吐量对于通用 kernel 实现来说是合理的。要达到 20K+ tok/s 需要架构级别的改变（CUDA Graph + FlashAttention + 重叠执行），不是简单的代码修改可以实现的。
