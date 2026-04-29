#!/usr/bin/env python3
"""Simple baseline test without DFlash"""
import sys
import os
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

os.environ['LMDEPLOY_SKIP_WARMUP'] = '1'

if __name__ == '__main__':
    import time
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

    print("=" * 70)
    print("Baseline Inference Test (no DFlash)")
    print("=" * 70)

    target_model_path = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
    prompts = ["What is the capital of France?"]

    backend_config = PytorchEngineConfig(
        dtype='float16',
        cache_max_entry_count=0.5,
        eager_mode=True,
    )

    print(f"\nCreating pipeline...")
    print(f"Model: {target_model_path}")

    try:
        pipe = pipeline(
            target_model_path,
            backend_config=backend_config,
            log_level='INFO',
        )

        print(f"Pipeline created successfully")

        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*70}")
            print(f"Test {i}: {prompt[:50]}...")
            print(f"{'='*70}")

            start_time = time.time()
            output = pipe(
                prompt,
                GenerationConfig(
                    max_new_tokens=128,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                ),
            )

            elapsed = time.time() - start_time
            num_tokens = len(output.token_ids)

            print(f"Done")
            print(f"  Generated tokens: {num_tokens}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Speed: {num_tokens/elapsed:.1f} tokens/s")
            print(f"  Output preview: {output.text[:100]}...")

        print(f"\nTest completed!")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()