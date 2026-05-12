#!/usr/bin/env python3
"""
简单的 DFlash 测试：检查每个阶段
"""

import os
import sys

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

# 1. 测试是否能导入 lmdeploy
try:
    import torch
    print('[1] lmdeploy 导入成功')
except Exception as e:
    print(f'[1] lmdeploy 导入失败: {e}')
    sys.exit(1)

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"


# 2. 测试不使用 speculative_config 的 Pipeline 创建
print('\n[2] 测试不使用 speculative_config 的 Pipeline 创建...')
try:
    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tensor_parallel=1,
        cache_max_entry_count=0.2,
        quant_policy=8,
        session_len=16384,
    )

    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        log_level='INFO'
    )
    print('[2] Pipeline 创建成功')

    # 简单推理测试
    prompt = "Python 是什么？"
    print('[2] 测试推理...')
    response = pipe([{"role": "user", "content": prompt}],
                   GenerationConfig(max_new_tokens=16, do_sample=False))
    print(f'[2] 推理成功: {response.text}')

    del pipe

except Exception as e:
    print(f'[2] Pipeline 创建/推理失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)


# 3. 测试使用 speculative_config 但禁用 DFlash 加载
print('\n[3] 测试使用 speculative_config 但禁用 DFlash 加载...')
try:
    import lmdeploy.turbomind.turbomind as tm_module

    # 保存原始的 _load_dflash_model
    original_load = tm_module.TurboMind._load_dflash_model

    # 替换为占位函数
    def _dummy_load(self):
        print('[3] 跳过 _load_dflash_model()')

    tm_module.TurboMind._load_dflash_model = _dummy_load

    speculative_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
        quant_policy=0,
    )

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tensor_parallel=1,
        cache_max_entry_count=0.2,
        quant_policy=8,
        session_len=16384,
        speculative_config=speculative_config,
    )

    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        log_level='INFO'
    )
    print('[3] Pipeline 创建成功（跳过 DFlash 加载）')

    # 简单推理测试
    prompt = "Python 是什么？"
    print('[3] 测试推理...')
    response = pipe([{"role": "user", "content": prompt}],
                   GenerationConfig(max_new_tokens=16, do_sample=False))
    print(f'[3] 推理成功: {response.text}')

    del pipe

except Exception as e:
    print(f'[3] Pipeline 创建/推理失败: {e}')
    import traceback
    traceback.print_exc()


print('\n[测试完成]')
