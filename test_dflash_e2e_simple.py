#!/usr/bin/env python3
"""DFlash 端到端测试 - 验证 Qwen3.6-35B-A3B-AWQ + DFlash speculative decoding.

用法:
  # 基础测试 (TP=4, 8 speculative tokens)
  python test_dflash_e2e_simple.py

  # 自定义参数
  python test_dflash_e2e_simple.py --tp 4 --num-spec-tokens 4 --max-tokens 32
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 添加 lmdeploy 到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, TurbomindEngineConfig
from lmdeploy.messages import SpeculativeConfig, EngineConfig

# 模型路径
TARGET_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ'
DRAFT_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3.6-35B-A3B-DFlash'


def print_section(title: str):
    """打印分隔符."""
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def test_dflash_pytorch_backend(tp: int = 4,
                                 num_spec_tokens: int = 8,
                                 max_tokens: int = 64,
                                 prompt: str = '你好'):
    """测试 PyTorch 后端的 DFlash."""

    print_section('DFlash PyTorch 后端测试')

    print(f'\n配置:')
    print(f'  Target Model: {TARGET_MODEL}')
    print(f'  Draft Model:  {DRAFT_MODEL}')
    print(f'  TP:           {tp}')
    print(f'  Spec Tokens:  {num_spec_tokens}')
    print(f'  Max Tokens:   {max_tokens}')
    print(f'  Prompt:       {prompt}')

    # 配置 DFlash
    spec_config = SpeculativeConfig(
        method='dflash',
        model=DRAFT_MODEL,
        num_speculative_tokens=num_spec_tokens,
    )

    # PyTorch engine config
    engine_config = PytorchEngineConfig(
        tp=tp,
        session_len=8192,
        cache_max_entry_count=0.8,
    )

    print_section('加载模型')
    print(f'正在加载模型... (TP={tp})')

    start_load = time.time()
    try:
        pipe = pipeline(
            model_path=TARGET_MODEL,
            backend_config=engine_config,
            speculative_config=spec_config,
            log_level='INFO',
        )
        load_time = time.time() - start_load
        print(f'✓ 模型加载完成 ({load_time:.1f}秒)')
    except Exception as e:
        print(f'✗ 模型加载失败: {e}')
        import traceback
        traceback.print_exc()
        return False

    # 生成配置
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
    )

    print_section('生成测试')
    print(f'Prompt: {prompt}\n')
    print('开始生成...\n')

    start_gen = time.time()
    try:
        response = pipe([prompt], gen_config=gen_config)
        gen_time = time.time() - start_gen

        output_text = response[0]

        print_section('生成结果')
        print(output_text)
        print_section('性能统计')
        print(f'生成时间:     {gen_time:.2f} 秒')
        print(f'生成 tokens:  {max_tokens}')
        print(f'吞吐量:       {max_tokens / gen_time:.1f} tokens/秒')

        return True

    except Exception as e:
        print(f'✗ 生成失败: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_dflash_turbomind_backend(tp: int = 4,
                                    num_spec_tokens: int = 8,
                                    max_tokens: int = 64,
                                    prompt: str = '你好'):
    """测试 TurboMind 后端的 DFlash."""

    print_section('DFlash TurboMind 后端测试')

    print(f'\n配置:')
    print(f'  Target Model: {TARGET_MODEL}')
    print(f'  Draft Model:  {DRAFT_MODEL}')
    print(f'  TP:           {tp}')
    print(f'  Spec Tokens:  {num_spec_tokens}')
    print(f'  Max Tokens:   {max_tokens}')
    print(f'  Prompt:       {prompt}')

    # 配置 DFlash
    spec_config = SpeculativeConfig(
        method='dflash',
        model=DRAFT_MODEL,
        num_speculative_tokens=num_spec_tokens,
    )

    # TurboMind engine config
    engine_config = TurbomindEngineConfig(
        tp=tp,
        session_len=8192,
        cache_max_entry_count=0.8,
    )

    print_section('加载模型')
    print(f'正在加载模型... (TP={tp})')

    start_load = time.time()
    try:
        pipe = pipeline(
            model_path=TARGET_MODEL,
            backend_config=engine_config,
            speculative_config=spec_config,
            log_level='INFO',
        )
        load_time = time.time() - start_load
        print(f'✓ 模型加载完成 ({load_time:.1f}秒)')
    except Exception as e:
        print(f'✗ 模型加载失败: {e}')
        import traceback
        traceback.print_exc()
        print('\n注意: TurboMind 后端的 DFlash 集成尚未完成')
        print('建议使用 PyTorch 后端进行测试')
        return False

    # 生成配置
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
    )

    print_section('生成测试')
    print(f'Prompt: {prompt}\n')
    print('开始生成...\n')

    start_gen = time.time()
    try:
        response = pipe([prompt], gen_config=gen_config)
        gen_time = time.time() - start_gen

        output_text = response[0]

        print_section('生成结果')
        print(output_text)
        print_section('性能统计')
        print(f'生成时间:     {gen_time:.2f} 秒')
        print(f'生成 tokens:  {max_tokens}')
        print(f'吞吐量:       {max_tokens / gen_time:.1f} tokens/秒')

        return True

    except Exception as e:
        print(f'✗ 生成失败: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='DFlash 端到端测试')
    parser.add_argument('--backend',
                        type=str,
                        default='pytorch',
                        choices=['pytorch', 'turbomind'],
                        help='后端选择 (默认: pytorch)')
    parser.add_argument('--target-model', default=TARGET_MODEL, help='Target 模型路径')
    parser.add_argument('--draft-model', default=DRAFT_MODEL, help='Draft 模型路径')
    parser.add_argument('--tp', type=int, default=4, help='Tensor Parallelism')
    parser.add_argument('--num-spec-tokens', type=int, default=8, help='Speculative tokens 数量')
    parser.add_argument('--max-tokens', type=int, default=64, help='最大生成 tokens')
    parser.add_argument('--prompt', type=str, default='你好，请介绍一下你自己', help='测试提示词')

    args = parser.parse_args()

    print_section('DFlash 端到端测试')
    print(f'Backend: {args.backend}')

    if args.backend == 'pytorch':
        success = test_dflash_pytorch_backend(
            tp=args.tp,
            num_spec_tokens=args.num_spec_tokens,
            max_tokens=args.max_tokens,
            prompt=args.prompt,
        )
    else:  # turbomind
        success = test_dflash_turbomind_backend(
            tp=args.tp,
            num_spec_tokens=args.num_spec_tokens,
            max_tokens=args.max_tokens,
            prompt=args.prompt,
        )

    if success:
        print_section('测试完成 ✓')
        return 0
    else:
        print_section('测试失败 ✗')
        return 1


if __name__ == '__main__':
    sys.exit(main())
