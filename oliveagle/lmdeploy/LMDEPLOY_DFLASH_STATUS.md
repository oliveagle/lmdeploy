# lmdeploy + DFlash 实现状态总结

> **日期**: 2026-04-25
> **状态**: 代码已实现，但 Qwen3.5 模型存在兼容性问题

## 已完成的工作

### 1. DFlash 模型架构 (`lmdeploy/pytorch/models/dflash.py`)
- ✅ `DFlashAttention`: Q 来自 draft states，KV 来自 concat(target_hidden + draft_hidden)
- ✅ `DFlashDecoderLayer`: RMSNorm → DFlashAttention → residual → FFN → residual
- ✅ `DFlashModel`: embed_tokens + FC层 + layers + norm + rotary_emb
- ✅ `DFlashForCausalLM`: 完整集成 lmdeploy 的 CudaGraphMixin

### 2. DFlash Proposer (`lmdeploy/pytorch/spec_decode/proposers/dflash.py`)
- ✅ `DFlashProposer` 一次性并行生成所有 draft token
- ✅ 注册到 `SPEC_PROPOSERS` 模块

### 3. 配置支持
- ✅ `SpecDecodeConfig.from_config` 中加入 `'dflash'` 到 `no_caches` 列表
- ✅ `module_map.py` 注册 `DFlashForCausalLM`
- ✅ `spec_agent.py` 中 dflash 方法跳过自回归循环

### 4. Qwen3.5 模型支持
- ✅ 新增 `configurations/qwen3_5.py`: 处理嵌套 `text_config`
- ✅ 新增 `models/qwen3_5.py`: 处理 `language_model.` 权重前缀
- ✅ `module_map.py` 注册 `Qwen3_5ForConditionalGeneration`

## 当前问题

### Qwen3.5 模型架构不兼容

**问题**: Qwen3.5-9B-HF (带 vision) 使用 `linear_attn` 层，与 Qwen3 标准架构不同。

**错误**:
```
KeyError: 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight'
```

**原因**:
- lmdeploy 的 `Qwen3ForCausalLM` 期望标准 attention 层
- Qwen3.5-9B-HF 混合了 linear attention 和 full attention
- 权重命名不同 (`linear_attn` vs `self_attn`)

### 解决方案

**使用纯文本 Qwen3.5 模型**:
```bash
# Qwen3.5-9B-Text (无 vision，纯文本)
TARGET_MODEL="/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3.5-9B-Text"
```

或使用 **Qwen3 (非 3.5) + DFlash**:
```bash
# 需要 Qwen3-8B target model 权重
TARGET_MODEL="/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3-0___6B"
DRAFT_MODEL="/mnt/eaget-4tb/models/Qwen3-8B-DFlash"
```

## DFlash 模型兼容性

| Target Model | Draft Model | 状态 | 备注 |
|---------------|-------------|------|------|
| Qwen3-8B-HF | Qwen3-8B-DFlash | ⚠️ 权重缺失 | target 只有 tokenizer |
| Qwen3.5-9B-Text | Qwen3.5-9B-DFlash | ⚠️ 架构不同 | draft 是 qwen3，target 是 qwen3_5 |
| Qwen3.5-9B-Text | Qwen3-8B-DFlash | ❌ vocab_size 不同 | 151936 vs 248320 |
| Qwen3-0.6B | Qwen3-8B-DFlash | ❌ 尺寸不匹配 | 0.6B vs 8B |

## 推荐测试路径

### 路径 1: 使用 Qwen3.5-Text + 自定义 DFlash draft
需要训练 Qwen3.5 的 DFlash draft model (model_type = qwen3_5_text)

### 路径 2: 使用 Qwen3 + DFlash
需要完整的 Qwen3-8B target model 权重（当前目录只有 tokenizer）

### 路径 3: 使用 vLLM DFlash 实现
vLLM 已有完整的 DFlash 支持，可参考:
- `/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/vllm/vllm/model_executor/models/dflash.py`

## 下一步

1. **获取完整模型**: 下载 Qwen3-8B-HF 或 Qwen3.5-9B-Text 的完整权重
2. **训练/获取匹配的 DFlash draft**: 确保 draft 与 target 的 vocab_size 和 model_type 匹配
3. **端到端测试**: 运行完整的 baseline + DFlash 对比测试

## 文件清单

| 文件 | 操作 | 状态 |
|------|------|------|
| `lmdeploy/pytorch/models/dflash.py` | 新建 | ✅ 完成 |
| `lmdeploy/pytorch/spec_decode/proposers/dflash.py` | 新建 | ✅ 完成 |
| `lmdeploy/pytorch/spec_decode/proposers/__init__.py` | 修改 | ✅ 完成 |
| `lmdeploy/pytorch/models/module_map.py` | 修改 | ✅ 完成 |
| `lmdeploy/pytorch/config.py` | 修改 | ✅ 完成 |
| `lmdeploy/pytorch/spec_decode/spec_agent.py` | 修改 | ✅ 完成 |
| `lmdeploy/pytorch/configurations/qwen3_5.py` | 新建 | ✅ 完成 |
| `lmdeploy/pytorch/models/qwen3_5.py` | 新建 | ✅ 完成 |
| `lmdeploy/benchmark_dflash_lmdeploy.py` | 新建 | ✅ 完成 |
| `lmdeploy/test_dflash_support.py` | 新建 | ✅ 完成 |
