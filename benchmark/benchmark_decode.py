import json
import os
import time
from pathlib import Path

import fire
import numpy as np
from lmdeploy import pipeline
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig


def benchmark(model_path, share_gpt_path, downsample=100, backend='turbomind', tp=1, save_to='decode_result'):
    """Benchmark using ShareGPT data.

    Please download `ShareGPT_V3_unfiltered_cleaned_split.json` as data for this benchmark.
    """

    start = time.monotonic()
    with open(share_gpt_path) as f:
        content = json.load(f)

    texts = []
    for c in content:
        for cc in c['conversations']:
            texts.append(cc['value'])

    print(f'Parse json in {time.monotonic() - start} seconds.')

    # Downsample texts
    texts = texts[::downsample]
    num_prompts = len(texts)

    print(f'Number of prompts: {num_prompts}')
    print(f'Average length (chars): {np.mean([len(t) for t in texts]):.0f}')

    # Create backend config
    if backend == 'turbomind':
        engine_config = TurbomindEngineConfig(tp=tp, max_batch_size=num_prompts)
    else:
        engine_config = PytorchEngineConfig(tp=tp, max_batch_size=num_prompts)

    start = time.monotonic()
    # Init pipeline
    pipe = pipeline(model_path, backend_config=engine_config)

    pipe_start = time.monotonic()
    print(f'Pipeline initialized in {pipe_start - start:.1f} seconds.')

    # Process all prompts
    responses = pipe(texts)

    elapsed = time.monotonic() - start
    total_chars = sum(len(r.text) if hasattr(r, 'text') and r.text else 0 for r in responses)
    print(f'Decoded {total_chars} chars in {elapsed:.1f} seconds, '
          f'{total_chars / elapsed:.1f} chars/s.')
    print(f'Decoded {num_prompts} prompts in {elapsed:.1f} seconds, '
          f'{num_prompts / elapsed:.1f} requests/s.')

    # Save results
    os.makedirs(os.path.dirname(save_to) if os.path.dirname(save_to) else '.', exist_ok=True)
    json_path = Path(save_to).with_suffix('.json')

    results = []
    for text, resp in zip(texts[:5], responses[:5]):
        results.append({
            'prompt': text[:100] + '...',
            'response': resp.text[:200] if hasattr(resp, 'text') else str(resp),
        })

    # Save results to JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {json_path}')

    print('Sample results (first 5):')
    for r in results:
        print(f'  Prompt: {r["prompt"]}')
        print(f'  Response: {r["response"]}')
        print()


if __name__ == '__main__':
    fire.Fire(benchmark)

    # llama-2 on 1 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 175.3 seconds, 9012.821089984884 tokens/s.
    # Decoded 7022 prompts in 175.3 seconds, 40.067481648961376 requests/s.

    # llama-2 on 3 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 77.9 seconds, 20268.736076299527 tokens/s.
    # Decoded 7022 prompts in 77.9 seconds, 90.10688248180179 requests/s.

    # llama-2 on 8 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 55.2 seconds, 28630.35872677815 tokens/s.
    # Decoded 7022 prompts in 55.2 seconds, 127.27939026361929 requests/s.

    # llama-2 on 8 A100:
    # data = ShareGPT, downsample = 10
    # Decoded 15991314 tokens in 242.7 seconds, 65893.38488718234 tokens/s.
    # Decoded 70216 prompts in 242.7 seconds, 289.33018970413536 requests/s.

    # Above time all includes time for workers to load model.
