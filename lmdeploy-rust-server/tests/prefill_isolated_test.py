#!/usr/bin/env python3
"""每个测试都重新创建 pipeline，确保没有状态残留"""
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

def measure_one(pipe, prompt, target, runs=3):
    """单次测量"""
    for r in range(runs):
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
                if response.generate_token_len:
                    break
        if ttft:
            tps = target / ttft * 1000
            print(f"  Run {r}: TTFT={ttft*1000:.1f}ms, TPS={tps:.0f} tok/s, gen_len={response.generate_token_len}")

# 分别测试每个 context，每次都重新创建 pipeline
for label, target in [("1K", 1024), ("4K", 4096), ("8K", 8192)]:
    print(f"\n{'=' * 80}")
    print(f"测试 {label} (target={target} tokens) - 隔离模式")
    print(f"{'=' * 80}")
    
    print("创建 pipeline...")
    pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(
        session_len=16384, max_batch_size=1, tp=1, model_format='awq', cache_max_entry_count=0.4,
    ), log_level='ERROR')
    
    # Warmup
    print(f"Warmup...")
    for _ in range(2):
        wp = gen_prompt(1024)
        for _ in pipe.stream_infer(
            [wp],
            gen_config=GenerationConfig(max_new_tokens=10, temperature=0.7),
            do_preprocess=False, stream_response=True,
        ):
            pass
    print("Warmup OK")
    
    prompt = gen_prompt(target)
    measure_one(pipe, prompt, target, runs=3)
    print(f"关闭 pipeline")
    pipe.close()
