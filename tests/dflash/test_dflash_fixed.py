#!/usr/bin/env python3
"""
修复 DFlash 测试：增加 cache_max_entry_count 来避免线性状态内存不足
"""

import os
import sys

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

import torch
from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"


def main():
    print('=' * 60)
    print('DFlash 测试 - 修复内存问题')
    print('=' * 60)

    # 使用更大的 cache_max_entry_count
    speculative_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
        quant_policy=0,
    )

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tensor_parallel=1,
        cache_max_entry_count=0.4,  # 从 0.2 增加到 0.4
        quant_policy=8,
        session_len=16384,
    )

    print('创建 Pipeline...')
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='INFO'
    )
    print('✓ Pipeline 创建成功！\n')

    # 简单推理测试
    prompt = "Python 是什么？"
    print('测试推理...')
    response = pipe([{"role": "user", "content": prompt}],
                   GenerationConfig(max_new_tokens=32, do_sample=False))
    print(f'✓ 推理成功！回复: {response.text}')

    print('\n' + '=' * 60)
    print('✓ 所有测试完成！')
    print('=' * 60)

    del pipe


if __name__ == '__main__':
    main()
