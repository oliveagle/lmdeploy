#!/usr/bin/env python3
"""
DFlash 调试脚本 - 逐步验证每个阶段
"""
import os
import sys
import torch

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'DEBUG'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("=" * 60)
print("DFlash 调试测试")
print("=" * 60)
print(f"目标: {target_model}")
print(f"草稿: {draft_model}\n")

# 检查 CUDA 设备
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# 配置
speculative_config = SpeculativeConfig(
    method='dflash',
    model=draft_model,
    num_speculative_tokens=8,
    quant_policy=0,
)

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tensor_parallel=1,
    cache_max_entry_count=0.5,
    quant_policy=8,
    session_len=16384,
)

print("创建 Pipeline (这可能需要几分钟)...")
try:
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='DEBUG'
    )
    print("✓ Pipeline 创建成功！\n")
except Exception as e:
    print(f"✗ Pipeline 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("运行单个推理测试...")
try:
    gen_config = GenerationConfig(max_new_tokens=32, do_sample=False)
    resp = pipe(
        [{"role": "user", "content": "Hello"}],
        gen_config=gen_config,
        sequence_start=True,
        sequence_end=True,
        chat_template_kwargs={'enable_thinking': False}
    )
    print(f"✓ 推理成功！")
    print(f"输出: {resp.text}")
    print(f"输出 tokens: {resp.token_ids}")
except Exception as e:
    print(f"✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n清理...")
del pipe
print("完成！")
