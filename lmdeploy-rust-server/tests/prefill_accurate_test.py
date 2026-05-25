#!/usr/bin/env python3
"""精确复现历史 benchmark 的测试方法"""
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

def test_pipeline():
    """使用与历史 benchmark 完全相同的方法"""
    print("=" * 80)
    print("精确复现历史 benchmark 方法 (使用 pipeline.stream_infer)")
    print("=" * 80)
    
    # 使用与历史 benchmark 完全相同的配置
    print("\n[1/2] 创建 pipeline (session_len=16384, 默认 max_prefill_token_num=8192)...")
    pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(
        session_len=16384,  # 与历史相同
        max_batch_size=1,
        cache_block_seq_len=64,
        tp=1,
        enable_prefix_caching=False,
        model_format='awq',
        cache_max_entry_count=0.4,
    ), log_level='ERROR')
    print("OK")
    
    results = {}
    for label, target in [("1K", 1024), ("4K", 4096), ("8K", 8192)]:
        prompt = gen_prompt(target)
        print(f"\n[2/2] 测试 {label} ({target} tokens)...")
        
        # Warmup
        print(f"  Warmup (2 runs)...", end=" ", flush=True)
        warmup_prompt = gen_prompt(1024)
        for _ in range(2):
            for _ in pipe.stream_infer(
                [warmup_prompt],
                gen_config=GenerationConfig(max_new_tokens=1, temperature=0.7),
                do_preprocess=False,
                stream_response=True,
            ):
                pass
        print("OK")
        
        # 测量 TTFT (与历史完全相同的方法)
        print(f"  Measure (3 runs)...", end=" ", flush=True)
        ttfts = []
        for r in range(3):
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
                print(f"{ttfts[-1]:.0f}ms ", end="", flush=True)
        
        if ttfts:
            avg_ttft = sum(ttfts) / len(ttfts)
            tps = target / avg_ttft * 1000
            results[label] = {
                "tokens": target,
                "avg_ttft_ms": avg_ttft,
                "tps": tps,
                "raw": ttfts
            }
            print(f"\n  -> Avg TTFT: {avg_ttft:.2f}ms, TPS: {tps:.0f} tok/s")
    
    pipe.close()
    return results

if __name__ == "__main__":
    results = test_pipeline()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"  {label}: TTFT={r['avg_ttft_ms']:.2f}ms, TPS={r['tps']:.0f} tok/s")
    
    # 保存
    with open("/mnt/data/lmdeploy/lmdeploy-rust-server/tests/prefill_accurate_results.json", "w") as f:
        json.dump(results, f, indent=2)
