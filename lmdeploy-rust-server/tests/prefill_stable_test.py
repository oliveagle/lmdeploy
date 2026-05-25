#!/usr/bin/env python3
"""稳定的 prefill 性能测试"""
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

def test_stable():
    """稳定的 prefill 测试"""
    print("=" * 80)
    print("稳定 Prefill 性能测试 (使用 pipeline.stream_infer)")
    print("=" * 80)
    
    print("\n[1/2] 创建 pipeline...")
    pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(
        session_len=16384,
        max_batch_size=1,
        cache_block_seq_len=64,
        tp=1,
        enable_prefix_caching=False,
        model_format='awq',
        cache_max_entry_count=0.4,
    ), log_level='ERROR')
    print("OK")
    
    # 预热：确保 GPU 达到稳定状态
    print("\n[2/2] 预热和测试...")
    warmup_prompt = gen_prompt(4096)
    for _ in range(5):
        for _ in pipe.stream_infer(
            [warmup_prompt],
            gen_config=GenerationConfig(max_new_tokens=10, temperature=0.7),
            do_preprocess=False,
            stream_response=True,
        ):
            pass
    print("预热完成")
    
    results = {}
    for label, target in [("1K", 1024), ("4K", 4096), ("8K", 8192)]:
        prompt = gen_prompt(target)
        
        # 测量 5 次，取中间 3 次（去除首尾）
        ttfts = []
        for r in range(5):
            start = time.perf_counter()
            ttft = None
            for response in pipe.stream_infer(
                [prompt],
                gen_config=GenerationConfig(max_new_tokens=512, temperature=0.7, do_sample=False),
                do_preprocess=False,
                stream_response=True,
            ):
                if ttft is None and response.generate_token_len:
                    ttft = time.perf_counter() - start
                    break
            if ttft:
                ttfts.append(ttft * 1000)
        
        # 取中间 3 次
        if len(ttfts) >= 5:
            stable_ttfts = ttfts[1:4]
            avg_ttft = sum(stable_ttfts) / len(stable_ttfts)
            tps = target / avg_ttft * 1000
            results[label] = {
                "tokens": target,
                "avg_ttft_ms": avg_ttft,
                "tps": tps,
                "all_ttfts": ttfts
            }
            print(f"{label}: Avg TTFT={avg_ttft:.2f}ms, TPS={tps:.0f} tok/s (runs: {[f'{t:.0f}' for t in stable_ttfts]})")
    
    pipe.close()
    return results

if __name__ == "__main__":
    results = test_stable()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"  {label}: TTFT={r['avg_ttft_ms']:.2f}ms, TPS={r['tps']:.0f} tok/s")
    
    with open("/mnt/data/lmdeploy/lmdeploy-rust-server/tests/prefill_stable_results.json", "w") as f:
        json.dump(results, f, indent=2)
