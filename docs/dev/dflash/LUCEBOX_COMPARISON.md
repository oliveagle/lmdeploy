# LMDeploy DFlash 问题分析 - lucebox 对比

## lucebox DFlash 实现关键点

### 1. 非因果 Attention

```cpp
// lucebox/src/qwen3_dflash_graph.cpp:139
ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, 
    /*mask=*/nullptr,  // ← 非因果！
    scale, ...);
```

**关键**: `mask=nullptr` 表示所有 tokens 可以 attend 到所有其他 tokens

### 2. 架构设计

```
输入:
  - noise_embed: [hidden, q_len, 1]    # Q 来自 noise
  - target_hidden_cat: [5*hidden, ctx_len, 1]  # K/V 来自 5 层 target
  - positions_q: [ctx_len..ctx_len+q_len-1]
  - positions_k: [0..ctx_len+q_len-1]

每层:
  Q = wq @ noise_embed  # 只处理 draft tokens
  K_ctx = wk @ target_feat  # 来自 target
  K_draft = wk @ noise_embed  # 来自 noise
  K = concat[K_ctx, K_draft]  # 拼接
  V = concat[V_ctx, V_draft]
  
  attn = flash_attn_ext(Q, K, V, mask=null)  # 非因果
```

### 3. 无状态设计

- **没有 KV cache** - 每次都是完整计算
- 输入 token 数固定 (q_len = 8)
- Target hidden states 每次重新计算

### 4. DDTree 验证

- Tree-structured verification
- Budget = 22
- 恢复最后 30% 的加速

## LMDeploy 当前实现对比

### ✅ 已正确实现

1. **非因果 attention kernel** - `dflash_kernels.cu` 已实现
2. **Draft model 加载** - 权重正确加载
3. **DFlash 启用** - 日志显示 `DFlash enabled`

### ❌ 问题所在

1. **Speculative decoding 逻辑未执行**
   - 日志中没有 `[DFlash] Collected aux hidden state`
   - 日志中没有 `[DFlash] Output X accepted draft tokens`

2. **可能原因**
   - `aux_hidden_states_` 收集条件不满足
   - `selected_token_pos` 没有正确传递
   - Phase 不匹配
   - 验证后的 tokens 没有被实际使用

## 下一步行动

### 方案 1: 调试现有实现

添加更多日志到 `unified_decoder.cc`:
```cpp
TM_LOG_INFO("[DFlash] Forward: phase=%d, enable=%d, model=%p",
    phase, enable_dflash_, dflash_draft_model_);
TM_LOG_INFO("[DFlash] aux_hidden_states_.size()=%zu", aux_hidden_states_.size());
```

### 方案 2: 参考 lucebox 架构

重新设计 DFlash 集成：
1. **独立进程** - 像 lucebox 一样，draft model 在独立进程中
2. **无状态设计** - 移除 KV cache 依赖
3. **明确的数据流**:
   ```
   Target (32层) → 捕获 5 层 hidden states
       ↓
   Draft (5层) + noise → 生成 8 个 draft tokens
       ↓
   Target 验证 → 接受/拒绝
   ```

### 方案 3: 简化实现

暂时禁用复杂功能，只保留核心：
1. 移除 `selected_token_pos` 依赖
2. 强制收集 aux_hidden_states（不管 phase）
3. 每次推理都触发 DFlash

## 文件位置

- lucebox 实现: `/home/oliveagle/submodules/lucebox-hub/dflash/`
- LMDeploy 实现: `src/turbomind/models/llama/`
- 关键文件:
  - `unified_decoder.cc` - DFlash 逻辑调用
  - `DFlashDraftModel.cu` - Draft model 实现
  - `dflash_kernels.cu` - Attention kernel