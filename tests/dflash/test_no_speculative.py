#!/usr/bin/env python3
"""
简单测试：不使用 speculative 配置
"""

import os
import sys

# 确保我们从源代码目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"


def main():
    print('=' * 60)
    print('无 Speculative 配置测试')
    print('=' * 60)

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tensor_parallel=1,
        cache_max_entry_count=0.4,
        quant_policy=8,
        session_len=16384,
    )

    print('创建 Pipeline...')
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
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
