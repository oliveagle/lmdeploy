#!/usr/bin/env python3
"""
Python TurboMind Benchmark - 建立正确的性能基线
测量 1K/4K/8K context 的 prefill 性能
"""

import os
import sys
import time
import json

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

try:
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
    print("✓ LMDeploy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import lmdeploy: {e}")
    sys.exit(1)

# 模型路径
MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"

# 测试配置
WARMUP_RUNS = 2
MEASURE_RUNS = 3
OUTPUT_TOKENS = 512
TEST_CONTEXTS = [1024, 4096, 8192]


def generate_prompt(token_count: int) -> str:
    """生成约指定 token 数量的 prompt"""
    text = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, token_count // 10)
    return text * repeats


def run_benchmark():
    print("=" * 80)
    print("Python TurboMind Benchmark - Prefill 性能测试")
    print("=" * 80)
    print(f"\nModel: {MODEL}")

    # 检查模型路径
    if not os.path.exists(MODEL):
        print(f"✗ Model path does not exist: {MODEL}")
        # 尝试替代路径
        alt_model = "/mnt/data/models/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ"
        if os.path.exists(alt_model):
            print(f"  Using alternative path: {alt_model}")
            global MODEL
            MODEL = alt_model
        else:
            print("✗ No valid model path found")
            return

    try:
        print("\n[1/3] 创建 pipeline...", end=" ", flush=True)
        pipe = pipeline(
            MODEL,
            backend_config=TurbomindEngineConfig(
                session_len=16384,
                max_batch_size=1,
                cache_block_seq_len=64,
                tp=1,
                enable_prefix_caching=False,
                model_format='awq',
                cache_max_entry_count=0.4,
            ),
            log_level='ERROR',
        )
        print("OK")
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        return

    results = {}

    for ctx_tokens in TEST_CONTEXTS:
        label = f"{ctx_tokens // 1024}K"
        print(f"\n[2/3] 测试 {label} context ({ctx_tokens} tokens)...")

        prompt = generate_prompt(ctx_tokens)
        gen_cfg = GenerationConfig(
            max_new_tokens=OUTPUT_TOKENS,
            temperature=0.7,
            do_sample=False,
        )

        # Warmup
        print(f"  Warmup ({WARMUP_RUNS} runs)...", end=" ", flush=True)
        for _ in range(WARMUP_RUNS):
            token_count = 0
            for _ in pipe.stream_infer(
                [prompt],
                gen_config=gen_cfg,
                do_preprocess=False,
                stream_response=True,
            ):
                token_count += 1
                if token_count >= 2:
                    break
        print("OK")

        # 测量
        print(f"  Measure ({MEASURE_RUNS} runs)...", end=" ", flush=True)
        ttfts = []
        total_times = []
        output_counts = []

        for r in range(MEASURE_RUNS):
            start_time = time.perf_counter()
            ttft_time = None
            output_tokens = 0

            for response in pipe.stream_infer(
                [prompt],
                gen_config=gen_cfg,
                do_preprocess=False,
                stream_response=True,
            ):
                if ttft_time is None and response.generate_token_len:
                    ttft_time = time.perf_counter() - start_time

                if response.generate_token_len:
                    output_tokens = response.generate_token_len

            total_time = time.perf_counter() - start_time

            ttfts.append(ttft_time * 1000 if ttft_time else 0)
            total_times.append(total_time * 1000)
            output_counts.append(output_tokens)
            print(f"{ttfts[-1]:.0f}ms ", end="", flush=True)

        # 统计
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_total = sum(total_times) / len(total_times)
        avg_output = sum(output_counts) / len(output_counts)

        prefill_tps = (ctx_tokens / avg_ttft * 1000) if avg_ttft > 0 else 0
        decode_ms = avg_total - avg_ttft
        decode_tps = (avg_output / (decode_ms / 1000)) if decode_ms > 0 else 0
        avg_itl = (decode_ms / avg_output) if avg_output > 0 else 0

        results[label] = {
            "context_length": ctx_tokens,
            "output_tokens": OUTPUT_TOKENS,
            "ttft_ms_avg": avg_ttft,
            "total_time_ms_avg": avg_total,
            "prefill_speed_tps_avg": prefill_tps,
            "decode_speed_tps_avg": decode_tps,
            "avg_itl_ms_avg": avg_itl,
            "avg_ttft_min": min(ttfts),
            "avg_ttft_max": max(ttfts),
        }

        print(f"\n    TTFT: {avg_ttft:.1f}ms, Prefill: {prefill_tps:.0f} tok/s, Decode: {decode_tps:.1f} tok/s")

    # 关闭 pipeline
    try:
        pipe.close()
    except:
        pass

    # 总结
    print("\n" + "=" * 80)
    print("总结 - Python TurboMind Prefill 性能")
    print("=" * 80)

    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            print(f"  {label:>6}: TTFT={r['ttft_ms_avg']:>7.2f}ms, "
                  f"Prefill={r['prefill_speed_tps_avg']:>8.0f} tok/s, "
                  f"Decode={r['decode_speed_tps_avg']:>5.1f} tok/s")

    # 对比历史数据
    print("\n" + "=" * 80)
    print("与历史数据对比")
    print("=" * 80)

    historical = {
        "1K": {"ttft": 69.62, "prefill": 14827.2, "decode": 41.2},
        "4K": {"ttft": 122.06, "prefill": 33727.5, "decode": 41.0},
        "8K": {"ttft": 191.37, "prefill": 42875.0, "decode": 40.6},
    }

    print(f"\n{'Context':>8} | {'TTFT (ms)':>12} | {'Prefill (tok/s)':>18} | {'Decode (tok/s)':>15}")
    print("-" * 70)
    for label in ["1K", "4K", "8K"]:
        if label in results:
            r = results[label]
            h = historical.get(label, {})
            print(f"{label:>8} | 当前: {r['ttft_ms_avg']:>7.2f} | 当前: {r['prefill_speed_tps_avg']:>10.0f} | 当前: {r['decode_speed_tps_avg']:>7.1f}")
            if h:
                print(f"         | 历史: {h['ttft']:>7.2f} | 历史: {h['prefill']:>10.0f} | 历史: {h['decode']:>7.1f}")
                ttft_ratio = r['ttft_ms_avg'] / h['ttft'] if h['ttft'] > 0 else 0
                prefill_ratio = h['prefill'] / r['prefill_speed_tps_avg'] if r['prefill_speed_tps_avg'] > 0 else 0
                print(f"         | 差异: {ttft_ratio:>6.1f}x | 差异: {prefill_ratio:>6.1f}x")
            print()

    # 保存结果
    output = {
        "engine": "LMDeploy Python TurboMind",
        "model": MODEL,
        "date": time.time(),
        "results": results,
        "historical": historical,
    }

    out_path = Path(__file__).parent / "benchmark_python_new.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    run_benchmark()