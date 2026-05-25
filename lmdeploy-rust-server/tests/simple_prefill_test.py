#!/usr/bin/env python3
"""简单准确的 prefill 性能测试"""
import os, sys, time, json
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'
sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count):
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats

async def test():
    print("加载引擎和 tokenizer...")
    tm = TurboMind(model_path=MODEL, engine_config=TurbomindEngineConfig(
        session_len=32768, max_batch_size=1, tp=1, enable_prefix_caching=False,
        max_prefill_token_num=32768, model_format='awq', cache_max_entry_count=0.4,
    ))
    tok = Tokenizer(MODEL)
    inst = tm.create_instance()
    
    # Warmup
    print("Warmup...")
    warmup = tok.encode(gen_prompt(1024))
    for i in range(2):
        async for _ in inst.async_stream_infer(session_id=i, input_ids=warmup,
            gen_config=GenerationConfig(max_new_tokens=1), sequence_start=True, sequence_end=True):
            pass
    
    results = {}
    for label, target in [("1K", 1024), ("4K", 4096), ("8K", 8192)]:
        prompt = gen_prompt(target)
        input_ids = tok.encode(prompt)
        actual = len(input_ids)
        
        times = []
        for r in range(3):
            start = time.perf_counter()
            async for out in inst.async_stream_infer(
                session_id=r+100, input_ids=input_ids,
                gen_config=GenerationConfig(max_new_tokens=1),
                sequence_start=True, sequence_end=True):
                if out.status.value in (1, 2):
                    times.append((time.perf_counter() - start) * 1000)
                    break
        
        if times:
            avg_ms = sum(times) / len(times)
            tps = actual / avg_ms * 1000
            results[label] = {"tokens": actual, "avg_ms": avg_ms, "tps": tps, "raw": times}
            print(f"{label}: {avg_ms:.1f}ms -> {tps:.0f} tok/s")
    
    tm.close()
    return results

if __name__ == "__main__":
    import asyncio
    results = asyncio.run(test())
    with open("/mnt/data/lmdeploy/lmdeploy-rust-server/tests/simple_prefill_results.json", "w") as f:
        json.dump(results, f, indent=2)
