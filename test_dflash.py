#!/usr/bin/env python3
"""DFlash 集成测试 - 验证 DFlash speculative decoding 功能.

用法:
  # 测试 DFlash (默认: 4 GPU, 8 speculative tokens, 64 生成 tokens)
  source /home/oliveagle/venvs/lmdeploy/bin/activate
  python test_dflash.py

  # 自定义参数
  python test_dflash.py --tp 4 --max-tokens 128 --prompt "Hello, world!"
"""

import argparse
import time
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.messages import SpeculativeConfig

# 模型路径
TARGET_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ'
DRAFT_MODEL = '/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3.6-35B-A3B-DFlash'


def main():
    parser = argparse.ArgumentParser(description='DFlash 集成测试')
    parser.add_argument('--target-model', default=TARGET_MODEL)
    parser.add_argument('--draft-model', default=DRAFT_MODEL)
    parser.add_argument('--tp', type=int, default=4)
    parser.add_argument('--num-spec-tokens', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--prompt', type=str, default='你好')
    args = parser.parse_args()

    # 配置 DFlash
    spec_config = SpeculativeConfig(
        method='dflash',
        model=args.draft_model,
        num_speculative_tokens=args.num_spec_tokens,
    )

    engine_config = PytorchEngineConfig(
        tp=args.tp,
        session_len=8192,
        cache_max_entry_count=0.8,
    )

    print('=' * 80)
    print('DFlash Speculative Decoding 测试')
    print('=' * 80)
    print(f'Target Model: {args.target_model}')
    print(f'Draft Model:  {args.draft_model}')
    print(f'TP: {args.tp}')
    print(f'Num Speculative Tokens: {args.num_spec_tokens}')
    print(f'Max Tokens: {args.max_tokens}')
    print('=' * 80)

    # 加载模型
    print('\n正在加载模型...')
    pipe = pipeline(
        model_path=args.target_model,
        backend_config=engine_config,
        speculative_config=spec_config,
        log_level='INFO',
    )
    print('✓ 模型加载完成\n')

    # 生成
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        do_sample=False,
    )

    print(f'Prompt: {args.prompt}\n')
    print('开始生成...\n')

    start_time = time.time()
    response = pipe([args.prompt], gen_config=gen_config)
    elapsed = time.time() - start_time

    # 输出
    print('=' * 80)
    print('生成结果')
    print('=' * 80)
    print(response[0])
    print('=' * 80)
    print(f'\n生成时间: {elapsed:.2f} 秒')
    print(f'吞吐量: {args.max_tokens / elapsed:.1f} tokens/秒')


if __name__ == '__main__':
    main()
