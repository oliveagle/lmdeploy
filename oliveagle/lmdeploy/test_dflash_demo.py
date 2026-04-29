#!/usr/bin/env python3
"""DFlash natural language demo."""
import sys
import os
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

os.environ['LMDEPLOY_SKIP_WARMUP'] = '1'

from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from lmdeploy.messages import SpeculativeConfig

def print_separator(title=''):
    print('=' * 70)
    if title:
        print(title)
        print('=' * 70)

def main():
    target_model = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
    draft_model = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'

    print_separator('DFlash Speculative Decoding Demo')
    print(f'Target: {target_model}')
    print(f'Draft: {draft_model}')

    # Build configs
    spec_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
    )

    backend_config = PytorchEngineConfig(
        dtype='float16',
        cache_max_entry_count=0.5,
        eager_mode=True,
    )

    print('\nCreating pipeline...')
    try:
        pipe = pipeline(
            target_model,
            speculative_config=spec_config,
            backend_config=backend_config,
            log_level='WARNING',
        )
        print('✅ Pipeline created successfully!')
    except Exception as e:
        print(f'❌ Pipeline creation failed: {e}')
        import traceback
        traceback.print_exc()
        return

    # Test prompts
    prompts = [
        'Explain quantum computing in simple terms for a high school student.',
        'Write a short poem about artificial intelligence.',
        'What are the key differences between Python and JavaScript?',
    ]

    for prompt in prompts:
        print_separator(f'Prompt: {prompt}')

        try:
            import time
            start = time.time()
            output = pipe(
                prompt,
                GenerationConfig(
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                ),
            )
            elapsed = time.time() - start

            print(f'\nResponse:\n{output.text}')
            print_separator()
            print(f'Speed: {len(output.token_ids)/elapsed:.1f} tokens/s')
            print(f'Tokens: {len(output.token_ids)}, Time: {elapsed:.2f}s')
            print()

        except Exception as e:
            print(f'❌ Error: {e}')
            import traceback
            traceback.print_exc()
            print()

    print_separator('Demo complete!')

if __name__ == '__main__':
    main()
