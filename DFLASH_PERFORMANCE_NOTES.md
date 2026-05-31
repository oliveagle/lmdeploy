# DFlash 性能问题分析

## 问题：普通推理和 DFlash 推理耗时一样

### 可能的原因

#### 1. **输出太短**
- 当前 `max_new_tokens=256`，但实际输出可能只有 50-100 tokens
- Speculative decoding 主要在长序列生成时才有优势
- 建议：增加 `max_new_tokens` 到 512 或更多

#### 2. **Qwen3.5 的 Linear Attention**
- Qwen3.5 使用了 Linear Attention (非标准 Transformer)
- Linear Attention 可能不支持 speculative decoding 的优化
- 错误信息：`Linear attention only supports stateless requests`

#### 3. **Draft Model 质量**
- Draft model 的接受率可能较低
- 如果接受率 < 50%，加速效果会被抵消
- 当前配置：`num_speculative_tokens=8`

#### 4. **Greedy 采样模式**
- `do_sample=False` (greedy) 接受率应该较高
- 但可能模型输出本身就很确定性

### 如何验证 DFlash 是否生效？

#### 方法 1: 检查日志
DFlash 正常工作时，日志应该显示：
```
[DFlash] Draft tokens accepted: X/Y (Z%)
```

#### 方法 2: 长文本生成
使用更长的输出：
```python
gen_config = GenerationConfig(
    max_new_tokens=1024,  # 增加到 1024
    do_sample=False,
)
```

#### 方法 3: 复杂 Prompt
使用需要更多推理的 prompt：
```python
prompt = "请详细介绍人工智能的发展历史，包括图灵测试、专家系统、深度学习等重要里程碑。"
```

### 已知限制

根据 `DFLASH_COMPARISON.md`：
- lucebox 的 DFlash 实现使用 **非因果 attention**，性能更好
- LMDeploy 当前使用 **因果 attention**，限制了 DFlash 效果
- 这可能是性能差异的主要原因

### 下一步

1. 尝试增加输出长度
2. 检查日志中的 accept rate
3. 考虑优化 attention 实现（需要 C++ 修改）