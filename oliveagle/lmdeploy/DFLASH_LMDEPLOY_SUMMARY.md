# DFlash lmdeploy 集成测试总结

> **测试时间**: 2026-04-27
> **测试环境**: Tesla V100 (sm_70), PyTorch 2.5.1+cu124

## 核心测试结果 ✅

```
[测试1] 直接加载 DFlash draft model...
✅ Config 加载成功
✅ DFlash 模型创建成功 (empty_init)
   - 参数数量: 3082.9M

[测试2] 检查模型权重...
✅ 模型权重存在: 2.0 GB

[测试3] 模拟 target model hidden states...
✅ Target hidden states 创建成功
   - target_layer_ids: [1, 8, 15, 22, 29]
   - num_target_layers: 5
   - target_hidden_size: 20480

[测试4] DFlashAttention forward pass (方案1核心)...
✅ DFlashAttention forward 成功
   - input_ids shape: torch.Size([1, 8])
   - hidden_states shape: torch.Size([1, 8, 4096])
   - logits shape: torch.Size([1, 8, 248320])
   - draft_token_ids shape: torch.Size([1, 8])

[测试5] 性能测试...
✅ 100 次迭代耗时: 0.662s
   - 平均每次: 6.622ms
   - 吞吐量: 1208 tokens/sec
```

## 实现方案

lmdeploy DFlash 使用 **`flash_attn_varlen_func`** 实现 non-causal attention：

```python
# DFlashAttention.forward (lmdeploy/pytorch/models/dflash.py:186)
attn_output = flash_attn_varlen_func(
    q_states, k_states, v_states,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=self.scaling,
    causal=False,  # 关键: non-causal
    kv_layout='hsd',
)
```

**关键设计**：
- Q 来自 draft hidden states
- K/V 来自 `target_hidden + draft_hidden` 的 element-wise 相加
- Non-causal: 所有 draft tokens 可以相互 attend
- 一次 forward pass 并行生成所有 draft tokens

## 已完成组件

| 组件 | 文件 | 状态 |
|------|------|------|
| DFlash Model | `lmdeploy/pytorch/models/dflash.py` | ✅ 完成 |
| DFlash Proposer | `lmdeploy/pytorch/spec_decode/proposers/dflash.py` | ✅ 完成 |
| Config 注册 | `lmdeploy/pytorch/configurations/dflash.py` | ✅ 完成 |

## 下一步工作

1. **Transformers 版本兼容** - 当前环境 `qwen3_5` 架构不被识别
2. **权重加载测试** - 需要足够 GPU 内存加载完整权重
3. **End-to-End pipeline** - 与 target model 协同测试
4. **Accept rate 测试** - 实际推理性能评估

## 性能指标

- **Draft 生成速度**: ~6.6ms/iter (8 tokens)
- **吞吐量**: ~1208 tokens/sec (V100)
- **理论加速**: 8x 并行生成 (vs 自回归)

