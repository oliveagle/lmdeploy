# DFlash 实现对比分析

## lucebox-hub/dflash vs LMDeploy Turbomind DFlash

### 架构对比

| 特性 | lucebox-hub/dflash | LMDeploy Turbomind |
|------|-------------------|-------------------|
| **运行时** | ggml (llama.cpp fork) | Turbomind (自定义 C++) |
| **模型格式** | GGUF Q4_K_M | AWQ / BF16 |
| **目标模型** | Qwen3.5-27B | Qwen3.6-35B-A3B |
| **Draft 层数** | 5 层 | 5 层 |
| **硬件目标** | RTX 3090 (24GB) | V100/A100 (16GB+) |
| **性能** | 129 tok/s | 待测试 |

### 关键设计差异

#### 1. 权重格式
**lucebox**:
- GGUF Q4_K_M (~16 GB)
- CPU 端 embedding (mmap)
- 不需要上传完整 tok_embd 到 GPU

**LMDeploy**:
- AWQ INT4 量化
- 完整权重在 GPU
- 需要 18GB+/GPU (TP=4)

#### 2. Attention 实现
**lucebox**:
- 混合架构: Full Attention (每 4 层) + Gated DeltaNet (其他层)
- Flash Attention 滑动窗口 (2048)
- 支持 128K context

**LMDeploy**:
- 标准 Transformer attention
- 无 DeltaNet
- 4K context

#### 3. Draft 模型结构
**lucebox**:
```cpp
// 5 层非因果 Qwen3 风格 block-diffusion
- fc @ target_hidden_cat → target_feat
- 每层: non-causal attention over (target_feat + noise)
- 共享 target 的 lm_head
```

**LMDeploy**:
```cpp
// 5 层简化 Qwen attention
- 收集指定层的 hidden states
- 标准 causal attention
- 独立 lm_head (或共享)
```

#### 4. KV Cache
**lucebox**:
- TQ3_0 量化 (3.5 bpv)
- 滑动 target_feat ring (4096 slots)
- 支持非对称 K/V 量化

**LMDeploy**:
- TurboQuant KV 量化
- 标准 KV cache
- 无滑动窗口

### 性能优化技术

#### lucebox 使用的优化:
1. **DDTree** (Tree-structured verify) - budget=22
2. **Flash Attention 滑动窗口** - FA window=2048
3. **KV 量化** - TQ3_0 (3.5 bpv)
4. **CPU embedding** - 不占用 GPU
5. **Prefix snapshot** - 跨请求共享

#### LMDeploy 可应用的优化:
1. ✅ KV 量化 (TurboQuant)
2. ⚠️ Flash Attention (需要集成)
3. ❌ DDTree (未实现)
4. ❌ 滑动窗口 (未实现)
5. ❌ CPU embedding (未实现)

### 建议

#### 短期 (立即可做):
1. **参考 lucebox 的 draft graph 结构**
   - 非 causal attention (mask=null)
   - Q 来自 noise, K/V 来自 (target_feat + noise)
   - RoPE (NEOX, theta=10M)

2. **简化 draft 模型**
   - 5 层就够了
   - 不需要复杂的 DeltaNet
   - 标准 Qwen attention 即可

3. **优化内存使用**
   - 考虑 CPU embedding
   - 使用更激进的 KV 量化
   - 草稿模型使用 Q4 量化

#### 中期 (需要重构):
1. **添加 Flash Attention 滑动窗口**
2. **实现 DDTree verification**
3. **支持更长 context**

#### 长期 (架构变更):
1. **支持混合架构** (Full Attention + DeltaNet)
2. **Prefix caching**
3. **跨请求共享**

### 代码参考

#### lucebox draft graph 关键部分:
```cpp
// 1. Feature fusion
ggml_tensor * target_feat = ggml_mul_mat(ctx, w.fc, in.target_hidden_cat);
target_feat = ggml_rms_norm(ctx, target_feat, eps);
target_feat = ggml_mul(ctx, target_feat, w.hidden_norm);

// 2. Q from noise only
ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, hn);
Q = ggml_rms_norm(ctx, Q, eps);

// 3. K and V from target_feat AND noise
ggml_tensor * Kctx = ggml_mul_mat(ctx, L.wk, target_feat);
ggml_tensor * Kn   = ggml_mul_mat(ctx, L.wk, hn);
ggml_tensor * K = ggml_concat(ctx, Kctx, Kn, 1);

// 4. Non-causal attention
ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
```

#### LMDeploy 当前实现:
```cpp
// 已在 unified_decoder.cc 中实现
- 收集 aux_hidden_states (指定层)
- DFlashDraftModel::GenerateDraft()
- DFlashDraftModel::VerifyDraft()
```

### 下一步行动

1. **验证当前 DFlash 实现** 是否与 lucebox 一致
2. **添加非 causal attention** (如果缺失)
3. **优化内存使用** 以支持 24GB GPU
4. **性能测试** 对比 lucebox 的 129 tok/s
