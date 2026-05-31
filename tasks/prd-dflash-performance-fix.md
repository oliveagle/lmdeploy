# PRD: DFlash Speculative Decoding 性能修复

## Overview

修复 LMDeploy TurboMind 后端中 DFlash speculative decoding 的性能问题。当前 DFlash 开启后性能下降（38 tok/s vs 基线 45 tok/s），且缺少接受率统计指标。目标是达到 80 tok/s 并正确报告 DFlash 统计。

## Goals

- **G-1**: DFlash 在 decode 阶段正确运行并验证 draft tokens
- **G-2**: DFlash 接受率正常计算并通过 Python API 可访问
- **G-3**: 开启 DFlash 后性能达到 80 tok/s
- **G-4**: 添加完整的 DFlash 统计指标（draft tokens, accepted, rejected, 接受率, 加速比）

## Quality Gates

这些命令必须通过：

- **QG-1**: `python3 setup.py build_ext --inplace` - C++ 编译成功
- **QG-2**: `pytest tests/test_lmdeploy/` - 所有单元测试通过
- **QG-3**: `python benchmark_dflash.py` - DFlash 开启时性能 >= 80 tok/s

---

## User Stories

### US-001: 修复 DFlash Decode 阶段执行
**Description:** 作为开发者，我需要 DFlash 在 decode 阶段（global_token_num == 1）正确执行，以确认 speculative decoding 逻辑正确。

**Acceptance Criteria:**
- [ ] 日志中出现 `[DFlash] === DECODE MODE: Verifying draft tokens ===`
- [ ] 日志中出现 `[DFlash] Found stored draft tokens`
- [ ] 日志中出现 `[DFlash] VerifyDraft: accepted=X tokens`
- [ ] 日志中出现 `[DFlash] Using X accepted tokens from DFlash`
- [ ] 确认 generation.cc 中的 `dflash_accepted_tokens` 检查逻辑被执行

**Files:**
- `src/turbomind/models/llama/unified_decoder.cc`
- `src/turbomind/generation/generation.cc`

---

### US-002: 添加完整 DFlash 统计指标
**Description:** 作为开发者，我需要完整的 DFlash 统计指标来监控 speculative decoding 效果。

**Acceptance Criteria:**
- [ ] Python API `tm.get_dflash_stats(0)` 返回包含以下字段的字典:
  - `total_draft_steps` - 总 draft 步数
  - `total_draft_tokens` - 总 draft tokens 数
  - `total_accepted_tokens` - 总接受 tokens 数
  - `total_rejected_tokens` - 总拒绝 tokens 数
  - `accept_rate` - 接受率 (float, 0-1)
  - `speedup_ratio` - 加速比 (相对于基线 45 tok/s)
- [ ] C++ 层正确追踪 `dflash_total_rejected_tokens_`
- [ ] `GetDFlashStats` 方法签名包含 rejected_tokens 参数

**Files:**
- `src/turbomind/generation/generation.h/cc`
- `src/turbomind/models/language_model.h/cc`
- `src/turbomind/engine/engine.h/cc`
- `src/turbomind/turbomind.h/cc`
- `src/turbomind/python/bind.cpp`

---

### US-003: 验证 DFlash 对性能的影响
**Description:** 作为开发者，我需要验证 DFlash 是否正确加速了解码过程。

**Acceptance Criteria:**
- [ ] 创建对比测试，确认 DFlash 开启/关闭的性能差异
- [ ] DFlash 关闭时：~45 tok/s
- [ ] DFlash 开启时：>= 80 tok/s
- [ ] 日志显示每个 request 的 DFlash 统计

**Files:**
- `benchmark_dflash.py`
- `test_dflash_fix.py`

---

### US-004: 优化 DFlash 接受率
**Description:** 作为开发者，我需要优化 DFlash 的接受率以达到 80 tok/s 目标。

**Acceptance Criteria:**
- [ ] 分析影响接受率的因素（draft model 质量、验证算法、hidden states 匹配度）
- [ ] 根据接受率调整 `num_speculative_tokens` 参数
- [ ] 记录接受率 >= 60% 时的性能指标
- [ ] 确认达到 80 tok/s 性能目标

**Files:**
- `src/turbomind/models/llama/DFlashDraftModel.cu`
- `src/turbomind/models/llama/unified_decoder.cc`

---

## Functional Requirements

### FR-1: DFlash 正确性
- FR-1.1: `global_token_num == 1` 时进入 DECODE MODE
- FR-1.2: 从 `args.try_("dflash_stored_draft_tokens")` 获取已存储的 draft tokens
- FR-1.3: 调用 `dflash_draft_model_->VerifyDraft()` 验证 drafts
- FR-1.4: 将 accepted tokens 输出到 `env["dflash_accepted_tokens"]`
- FR-1.5: Generation 模块检查 `env.contains("dflash_accepted_tokens")` 并使用 accepted tokens

### FR-2: 统计追踪
- FR-2.1: `dflash_total_draft_tokens_` 记录每步尝试的 draft tokens 数
- FR-2.2: `dflash_total_accepted_tokens_` 记录接受的 tokens 数
- FR-2.3: `dflash_total_rejected_tokens_` 记录拒绝的 tokens 数
- FR-2.4: 计算 accept_rate = accepted / draft_tokens
- FR-2.5: 计算 speedup_ratio = 当前性能 / 基线性能 (45 tok/s)

### FR-3: 日志输出
- FR-3.1: 清晰的阶段切换日志（DECODE MODE / PREFILL MODE）
- FR-3.2: draft tokens 生成和验证结果的日志
- FR-3.3: 统计信息汇总日志

## Non-Goals

- 不修改 DFlash draft model 的训练代码
- 不修改 CUDA kernel 实现细节
- 不支持多 GPU 场景的 DFlash（当前 TP=1）
- 不修改 vLLM 或其他后端的 DFlash 实现

## Technical Considerations

### 架构链路
```
Python API (bind.cpp)
  └── TurboMind::GetDFlashStats
        └── Engine::GetDFlashStats
              └── LanguageModel::GetDFlashStats
                    └── Generation::GetDFlashStats
                          └── impl_->dflash_total_*_ (statistics)
```

### 关键文件位置
- DFlash 主逻辑: `src/turbomind/models/llama/unified_decoder.cc`
- 统计收集: `src/turbomind/generation/generation.cc`
- Python 绑定: `src/turbomind/python/bind.cpp`

### 已知问题
- 当前 `phase` 变量表示异步 pipeline 阶段，不是 prefill/decode 标志
- 使用 `global_token_num` 判断当前阶段更可靠
- `aux_hidden_states_` 需要在每个 Forward 迭代开始时清空

## Success Metrics

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Decode 速度 | 38 tok/s | >= 80 tok/s |
| 接受率 | N/A | >= 60% |
| DFlash 统计完整性 | 部分 | 100% |
| 单元测试通过率 | 未知 | 100% |

## Open Questions

- Q1: 接受率低于预期时，是否需要调整 `num_speculative_tokens` 参数？
- Q2: 是否需要添加 draft tokens 缓存机制？
- Q3: 当前 VerifyDraft 使用的阈值是否需要调整？