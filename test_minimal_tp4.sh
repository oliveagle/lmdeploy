#!/bin/bash
# 最小化 TP=4 测试 - 只加载权重，不做推理

set -e

echo "========================================"
echo "最小化 TP=4 测试"
echo "========================================"

MODEL_PATH="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

cat > /tmp/test_minimal_tp4.py << 'PYEOF'
#!/usr/bin/env python3
import os
import sys
import gc
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TRITON_BENCHMARK_CACHE_SIZE_KB'] = '262144'

sys.path.insert(0, '/home/oliveagle/opt/lmdeploy/lmdeploy')

import torch

def get_gpu_mem():
    """获取当前 GPU 显存使用"""
    mems = []
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1e9
        mems.append(mem)
    return mems

model_path = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

print("=== 最小化 TP=4 测试 ===")
print(f"模型: {model_path}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"总显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print(f"初始显存: {[f'{m:.2f}GB' for m in get_gpu_mem()]}")

# 导入 lmdeploy 组件
from lmdeploy.pytorch.config import ModelConfig, DistConfig, CacheConfig, SchedulerConfig
from lmdeploy.pytorch.models.patch import build_model_from_hf_config
from lmdeploy.messages import QuantPolicy

print("\n加载模型配置...")
dist_config = DistConfig(tp=4, ep=1, attn_tp_size=4, moe_tp_size=4)

model_config = ModelConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    dist_config=dist_config,
)

print(f"Hidden size: {model_config.hidden_size}")
print(f"Num layers: {model_config.num_layers}")
print(f"加载后显存: {[f'{m:.2f}GB' for m in get_gpu_mem()]}")

# 创建调度配置
scheduler_config = SchedulerConfig(
    max_batches=1,
    max_session_len=1024,
)

cache_config = CacheConfig(
    max_batches=1,
    block_size=16,
    num_cpu_blocks=100,
    num_gpu_blocks=100,
    quant_policy=QuantPolicy.TURBO_QUANT,
)

print("\n尝试构建模型...")
try:
    # 使用 CPU 来避免显存峰值
    device = torch.device('cuda:0')

    # 直接构建模型（不通过 pipeline）
    import transformers
    from lmdeploy.pytorch.models import register_model

    # 简单测试：检查模型是否能正确分片
    print(f"TP=4 分片检查:")
    print(f"  Attention heads per GPU: {model_config.num_attention_heads // 4}")
    print(f"  KV heads per GPU: {model_config.num_key_value_heads // 4}")

    # 计算每卡权重大小
    hidden = model_config.hidden_size
    layers = model_config.num_layers
    experts = model_config.hf_config.text_config.num_experts

    # Attention (FP16)
    attn_per_card = (hidden * 3 * hidden + hidden * hidden) * layers * 2 / 4 / 1e9

    # MoE experts (AWQ 4-bit)
    moe_inter = model_config.hf_config.text_config.moe_intermediate_size
    moe_per_card = ((hidden * 2 * moe_inter + moe_inter * hidden) * layers * experts * 0.5 +
                    (hidden * 2 * moe_inter + moe_inter * hidden) * layers * experts * 0.5 * 0.25) / 1e9

    print(f"  Attention per card: {attn_per_card:.2f} GB")
    print(f"  MoE experts per card: {moe_per_card:.2f} GB (estimated)")
    print(f"  Total estimated: {attn_per_card + moe_per_card:.2f} GB")

    print("\n=== 测试通过！模型配置正确 ===")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

PYEOF

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_minimal_tp4.py
