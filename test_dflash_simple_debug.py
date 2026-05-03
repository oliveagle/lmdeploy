#!/usr/bin/env python3
"""DFlash 集成简单测试 - 使用更小的配置来快速验证功能"""

import sys
sys.path.insert(0, '.')

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.messages import SpeculativeConfig

# 模型路径
TARGET_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ'
DRAFT_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3.6-35B-A3B-DFlash'

def test_without_speculative():
    """首先测试不使用 speculative decoding，确保模型可以正常加载"""
    print('=' * 80)
    print('测试 1: 不使用 Speculative Decoding（基准测试）')
    print('=' * 80)

    engine_config = PytorchEngineConfig(
        tp=1,  # 先用 TP=1 测试
        session_len=2048,
        cache_max_entry_count=0.5,
    )

    print(f'\n正在加载模型 (TP=1)...')
    try:
        pipe = pipeline(
            model_path=TARGET_MODEL,
            backend_config=engine_config,
            log_level='INFO',
        )
        print('✓ 模型加载成功')

        # 简单生成测试
        gen_config = GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
        )

        prompt = '你好'
        print(f'\nPrompt: {prompt}')
        response = pipe([prompt], gen_config=gen_config)
        print(f'Response: {response[0]}')
        print('\n✓ 基准测试通过')
        return True
    except Exception as e:
        print(f'✗ 基准测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_with_dflash():
    """测试使用 DFlash speculative decoding"""
    print('\n' + '=' * 80)
    print('测试 2: 使用 DFlash Speculative Decoding')
    print('=' * 80)

    spec_config = SpeculativeConfig(
        method='dflash',
        model=DRAFT_MODEL,
        num_speculative_tokens=4,
    )

    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=2048,
        cache_max_entry_count=0.5,
    )

    print(f'\n正在加载模型 (TP=1, DFlash)...')
    try:
        pipe = pipeline(
            model_path=TARGET_MODEL,
            backend_config=engine_config,
            speculative_config=spec_config,
            log_level='INFO',
        )
        print('✓ 模型加载成功')

        gen_config = GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
        )

        prompt = '你好'
        print(f'\nPrompt: {prompt}')
        response = pipe([prompt], gen_config=gen_config)
        print(f'Response: {response[0]}')
        print('\n✓ DFlash 测试通过')
        return True
    except Exception as e:
        print(f'✗ DFlash 测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    print('\n' + '=' * 80)
    print('DFlash 集成测试脚本')
    print('=' * 80)
    print(f'Target Model: {TARGET_MODEL}')
    print(f'Draft Model:  {DRAFT_MODEL}')
    print('=' * 80)

    # 测试 1: 基准测试（不使用 speculative）
    baseline_ok = test_without_speculative()

    # 测试 2: DFlash 测试
    dflash_ok = test_with_dflash()

    print('\n' + '=' * 80)
    print('测试总结')
    print('=' * 80)
    print(f'基准测试: {"✓ 通过" if baseline_ok else "✗ 失败"}')
    print(f'DFlash测试: {"✓ 通过" if dflash_ok else "✗ 失败"}')

    if baseline_ok and dflash_ok:
        print('\n✓ 所有测试通过！DFlash 集成成功')
        return 0
    else:
        print('\n✗ 部分测试失败，需要进一步调试')
        return 1

if __name__ == '__main__':
    sys.exit(main())
