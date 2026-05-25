#!/usr/bin/env python3
import os, sys, time, json
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'
sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count):
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats

pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(
    session_len=16384, max_batch_size=1, tp=1, model_format='awq', cache_max_entry_count=0.4,
), log_level='ERROR')

# Warmup
for _ in range(3):
    for _ in pipe.stream_infer([gen_prompt(1024)], gen_config=GenerationConfig(max_new_tokens=10, temperature=0.7), do_preprocess=False, stream_response=True):
        pass

print("Testing 1K...")
for r in range(3):
    prompt = gen_prompt(1024)
    start = time.perf_counter()
    ttft = None
    for response in pipe.stream_infer(
        [prompt],
        gen_config=GenerationConfig(max_new_tokens=512, temperature=0.7, do_sample=False),
        do_preprocess=False,
        stream_response=True,
    ):
        if ttft is None:
            ttft = time.perf_counter() - start
            print(f"  Run {r}: TTFT={ttft*1000:.2f}ms, generate_token_len={response.generate_token_len}")
            break

print("Testing 4K...")
for r in range(3):
    prompt = gen_prompt(4096)
    start = time.perf_counter()
    ttft = None
    for response in pipe.stream_infer(
        [prompt],
        gen_config=GenerationConfig(max_new_tokens=512, temperature=0.7, do_sample=False),
        do_preprocess=False,
        stream_response=True,
    ):
        if ttft is None:
            ttft = time.perf_counter() - start
            print(f"  Run {r}: TTFT={ttft*1000:.2f}ms, generate_token_len={response.generate_token_len}")
            break

print("Testing 8K...")
for r in range(3):
    prompt = gen_prompt(8192)
    start = time.perf_counter()
    ttft = None
    for response in pipe.stream_infer(
        [prompt],
        gen_config=GenerationConfig(max_new_tokens=512, temperature=0.7, do_sample=False),
        do_preprocess=False,
        stream_response=True,
    ):
        if ttft is None:
            ttft = time.perf_counter() - start
            print(f"  Run {r}: TTFT={ttft*1000:.2f}ms, generate_token_len={response.generate_token_len}")
            break

pipe.close()
