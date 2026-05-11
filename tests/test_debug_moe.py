#!/usr/bin/env python3
"""
调试 AWQ MoE 的 topk_ids 问题
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.messages import QuantPolicy

# Monkey patch 添加调试代码
import lmdeploy.pytorch.nn.moe.awq as awq_module

original_gemm = awq_module.FusedMoEAWQ.gemm

def patched_gemm(self, state: dict):
    """添加调试的 gemm 方法"""
    from lmdeploy.pytorch.nn.moe.base import MoeType
    moe_type = state['moe_type']

    if moe_type == MoeType.Default or (isinstance(moe_type, str) and moe_type == 'Default'):
        hidden_states = state['hidden_states']
        topk_weights = state['topk_weights']
        topk_ids = state['topk_idx']

        print(f"[DEBUG] gemm called:")
        print(f"  hidden_states.shape={hidden_states.shape}")
        print(f"  topk_weights.shape={topk_weights.shape}")
        print(f"  topk_ids.shape={topk_ids.shape}")
        print(f"  topk_ids.min={topk_ids.min().item()}")
        print(f"  topk_ids.max={topk_ids.max().item()}")
        print(f"  self.num_experts={self.num_experts}")

        # 检查 topk_ids 是否超出范围
        max_expert_id = topk_ids.max().item()
        if max_expert_id >= self.num_experts:
            print(f"[ERROR] topk_ids 包含无效的 expert ID: {max_expert_id} >= {self.num_experts}")
            print(f"  这会导致后续处理出错！")
            # 修正无效的 ID
            topk_ids = torch.clamp(topk_ids, 0, self.num_experts - 1)
            print(f"  已修正 topk_ids 范围到 [0, {self.num_experts - 1}]")

        # 更新 state
        state['topk_idx'] = topk_ids

    return original_gemm(self, state)

awq_module.FusedMoEAWQ.gemm = patched_gemm

print("✓ 已应用调试补丁")

# 现在运行测试
MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

print("\n加载模型 (TP=4, TurboQuant)...")
engine_config = PytorchEngineConfig(
    tp=4,
    session_len=4096,
    cache_max_entry_count=0.5,
    max_batch_size=1,
    block_size=64,
    eager_mode=True,
    quant_policy=QuantPolicy.TURBO_QUANT,
    dtype='float16',
)

try:
    import time
    start = time.time()
    pipe = pipeline(
        model_path=MODEL_PATH,
        trust_remote_code=True,
        backend_config=engine_config,
    )
    load_time = time.time() - start
    print(f"✓ 模型加载成功 ({load_time:.2f}s)")

    print("\n测试推理...")
    response = pipe("你好", max_new_tokens=32, do_sample=False)
    print(f"输出: {response.text}")
    print("\n✓ 测试成功！")
    sys.exit(0)
except Exception as e:
    import traceback
    print(f"\n✗ 测试失败: {e}")
    traceback.print_exc()
    sys.exit(1)
