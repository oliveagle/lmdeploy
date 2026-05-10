# LMDeploy + DFlash + TurboQuant 测试总结

**测试时间**: 2026-05-10
**测试模型**: Qwen3.6-35B-A3B-AWQ
**Draft 模型**: Qwen3.6-35B-A3B-DFlash

## 测试环境

- **GPU**: 4x Tesla V100-SXM2-16GB (NVLink 互连)
- **Python**: 3.12
- **LMDeploy**: 0.12.3 (源码版本)
- **虚拟环境**: ~/venvs/lmdeploy

## 测试结果

### ❌ 测试 1: LMDeploy PyTorch Backend + TurboQuant

**状态**: 失败

**错误**: `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`

**错误位置**: `lmdeploy/pytorch/nn/moe/awq.py:309`

**根本原因**: AWQ MoE 实现中，`topk_ids` 张量包含无效的 expert ID，导致 `torch.unique(topk_ids).cpu().tolist()` 访问非法内存。

**堆栈跟踪**:
```
File "lmdeploy/pytorch/nn/moe/awq.py", line 309, in gemm
    unique_experts = torch.unique(topk_ids).cpu().tolist()
```

### ❌ 测试 2: LMDeploy Turbomind Backend + DFlash

**状态**: 失败

**错误 1**: `Fallback to pytorch engine because turbomind engine is not installed correctly`

**错误 2**: 与测试 1 相同的 CUDA 非法内存访问错误

## 问题分析

### 1. AWQ MoE Bug

Qwen3.6-35B-A3B-AWQ 模型有 256 个专家 × 40 层。LMDeploy 的 AWQ MoE 实现中存在 bug：

```python
# lmdeploy/pytorch/nn/moe/awq.py:309
unique_experts = torch.unique(topk_ids).cpu().tolist()
```

`topk_ids` 可能包含超出有效范围的 expert ID（比如 -1 或 >= num_experts），导致 CUDA 内存访问错误。

### 2. NCCL P2P 问题

使用 `CUDA_VISIBLE_DEVICES=1,2,3,4` 时，虽然都是 V100，但 NCCL 仍然报告 P2P 错误：
```
Cuda failure 217 'peer access is not supported between these two devices'
```

### 3. Turbomind 引擎未安装

```
Fallback to pytorch engine because turbomind engine is not installed correctly
```

## 建议修复

### 修复 1: AWQ MoE topk_ids 验证

在 `lmdeploy/pytorch/nn/moe/awq.py` 中添加验证：

```python
# 在 line 309 之前添加
# 验证 topk_ids 范围
if topk_ids.max() >= self.num_experts or topk_ids.min() < 0:
    # 修正无效的 expert ID
    topk_ids = torch.clamp(topk_ids, 0, self.num_experts - 1)

unique_experts = torch.unique(topk_ids).cpu().tolist()
```

### 修复 2: 使用正确的 GPU 选择

使用 `CUDA_VISIBLE_DEVICES=0,1,2,3` (物理 V100)，但确保 PyTorch 能正确识别设备。

### 修复 3: 安装 Turbomind 引擎

从源码编译 Turbomind：
```bash
cd /home/oliveagle/opt/lmdeploy/lmdeploy
pip install -e .
```

## 下一步

1. 修复 AWQ MoE bug
2. 重新编译 Turbomind 引擎
3. 测试基础 LMDeploy 功能
4. 测试 DFlash speculative decoding

## 相关文件

- 测试脚本: `test_turbomind_dflash.py`
- AWQ MoE 实现: `lmdeploy/pytorch/nn/moe/awq.py`
- Qwen3.5 MoE 模型: `lmdeploy/pytorch/models/qwen3_5_moe.py`
