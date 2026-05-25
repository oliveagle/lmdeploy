#!/usr/bin/env python3
"""TTFT 测量调试"""
import os, sys, time
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'
sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(
    session_len=16384, max_batch_size=1, tp=1, model_format='awq', cache_max_entry_count=0.4,
), log_level='ERROR')

# Warmup
for _ in range(2):
    for _ in pipe.stream_infer(["hello"], gen_config=GenerationConfig(max_new_tokens=5, temperature=0.7), do_preprocess=False, stream_response=True):
        pass

print("1K TTFT 调试...")
prompt = "The quick brown fox jumps over the lazy dog. " * 102
start = time.perf_counter()
count = 0
for response in pipe.stream_infer([prompt], gen_config=GenerationConfig(max_new_tokens=512, temperature=0.7, do_sample=False), do_preprocess=False, stream_response=True):
    count += 1
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Response #{count}: gen_token_len={response.generate_token_len}, text_len={len(response.text)}, elapsed={elapsed:.1f}ms")
    if count >= 5:
        break

pipe.close()
