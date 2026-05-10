# DFlash Speculative Decoding

> **状态**: ✅ 集成完成
> **集成日期**: 2026-05-04
> **最后更新**: 2026-05-07

## 概述

DFlash Speculative Decoding 是一种高效的推测解码方法，通过小型 draft 模型预测多个 token，然后由大型 target 模型并行验证，实现 1.7x-2.0x 的解码加速。

## 集成状态

**所有核心功能已集成到 LMDeploy PyTorch 后端** ✅

### 已完成的任务

| 任务 | 文件 | 状态 |
|------|------|------|
| DFlashDraftModel | `lmdeploy/pytorch/models/dflash.py` | ✅ |
| DFlashProposer | `lmdeploy/pytorch/spec_decode/proposers/dflash.py` | ✅ |
| SpeculativeAgent 修复 | `lmdeploy/pytorch/spec_decode/spec_agent.py` | ✅ |
| Qwen3.5 MoE AWQ 支持 | `lmdeploy/pytorch/models/qwen3_5_moe.py` | ✅ |
| 模块注册 | `lmdeploy/pytorch/models/module_map.py` | ✅ |

## 快速开始

### 最小示例

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.messages import SpeculativeConfig

TARGET_MODEL = "/path/to/Qwen3.6-35B-A3B-AWQ"
DRAFT_MODEL = "/path/to/Qwen3.6-35B-A3B-DFlash"

spec_config = SpeculativeConfig(
    method='dflash',
    model=DRAFT_MODEL,
    num_speculative_tokens=8,
)

engine_config = PytorchEngineConfig(
    tp=4,  # 推荐 TP4 或 TP2
    session_len=8192,
)

pipe = pipeline(
    model_path=TARGET_MODEL,
    backend_config=engine_config,
    speculative_config=spec_config,
)

response = pipe("你好，请介绍一下自己：")
print(response.text)
```

### 环境要求

**GPU**: A100, H100, L4, RTX 4090 或其他 sm_80+ 显卡

**原因**: 当前环境（V100 + CUDA 12.8）存在 TileLang 兼容性问题：
```
RuntimeError: No suitable user-defined conversion from "__nv_bfloat16" to "__half" exists
```

## 预期性能

| 指标 | 预期值 |
|------|--------|
| Decode 加速比 | 1.7x - 2.0x |
| Acceptance Rate | 60% - 80% |

## 相关文档

- [测试指南](test-guide.md) - 详细的测试步骤和脚本
- [V100 限制](v100-limitations.md) - V100 GPU 已知问题和限制

## 参考资源

- DFlash 论文: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2305.16398)
- 集成测试: `test_dflash_integration.py`
