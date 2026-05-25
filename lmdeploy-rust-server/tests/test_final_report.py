#!/usr/bin/env python3
"""最终性能报告"""
import json
from pathlib import Path

historical_data = {
    "engine": "LMDeploy Python TurboMind",
    "model": "Qwen3.6-35B-A3B-AWQ",
    "gpu": "Tesla V100 32GB (PG503-216)",
    "cuda": "12.5",
    "date": "2026-05-18",
    "lmdeploy_version": "0.13.0",
    "results": {
        "1K": {
            "context_length": 1024,
            "ttft_ms_avg": 69.62,
            "prefill_speed_tps_avg": 14827.2,
            "decode_speed_tps_avg": 41.2,
        },
        "4K": {
            "context_length": 4096,
            "ttft_ms_avg": 122.06,
            "prefill_speed_tps_avg": 33727.5,
            "decode_speed_tps_avg": 41.0,
        },
        "8K": {
            "context_length": 8192,
            "ttft_ms_avg": 191.37,
            "prefill_speed_tps_avg": 42875.0,
            "decode_speed_tps_avg": 40.6,
        }
    }
}

print("=" * 80)
print("最终性能报告")
print("=" * 80)
print()

print("=" * 80)
print("历史高性能 (2026-05-18)")
print("=" * 80)
for label in ["1K", "4K", "8K"]:
    r = historical_data["results"][label]
    print(f"  {label:>6}: TTFT={r['ttft_ms_avg']:>7.2f}ms, Prefill={r['prefill_speed_tps_avg']:>8.0f} tok/s, Decode={r['decode_speed_tps_avg']:>5.1f} tok/s")

print()
print("=" * 80)
print("Python pipeline (2026-05-26)")
print("=" * 80)
python_result_path = Path(__file__).parent / "prefill_correct.json"
if python_result_path.exists():
    with open(python_result_path) as f:
        data = json.load(f)
        for label in ["1K", "4K", "8K"]:
            if label in data.get("results", {}):
                r = data["results"][label]
                print(f"  {label:>6}: TTFT={r['ttft_ms_avg']:>7.2f}ms, Prefill={r['prefill_speed_tps_avg']:>8.0f} tok/s, Decode={r['decode_speed_tps_avg']:>5.1f} tok/s")
else:
    print("  无 Python 结果")

print()
print("=" * 80)
print("Rust Server (当前)")
print("=" * 80)
rust_result_path = Path(__file__).parent / "prefill_real.json"
if rust_result_path.exists():
    with open(rust_result_path) as f:
        data = json.load(f)
        for label in ["1K", "2K", "4K", "8K"]:
            if label in data.get("results", {}):
                r = data["results"][label]
                if "context_length" in r:
                    print(f"  {label:>6}: TTFT={r.get('avg_ms', 0):>7.1f}ms, Prefill={r.get('avg_tps', 0):>8.0f} tok/s")
                else:
                    print(f"  {label:>6}: TTFT={r.get('avg_ms', 0):>7.1f}ms, Prefill={r.get('avg_tps', 0):>8.0f} tok/s")
else:
    print("  无 Rust 结果")

print()
print("=" * 80)
print("Python vs Rust 对比 (当前)")
print("=" * 80)
python_data = {}
rust_data = {}
if python_result_path.exists():
    with open(python_result_path) as f:
        pd = json.load(f)
        python_data = pd.get("results", {})
if rust_result_path.exists():
    with open(rust_result_path) as f:
        rd = json.load(f)
        rust_data = rd.get("results", {})
print(f"{'Context':>8} | {'Python':>10} | {'Rust':>10} | {'Python/Rust':>12} | {'Rust/Python':>12}")
print("-" * 60)
for label in ["1K", "4K", "8K"]:
    p_tps = None
    r_tps = None
    if label in python_data:
        p_tps = python_data[label].get("prefill_speed_tps_avg") or python_data[label].get("avg_tps")
    if label in rust_data:
        r_tps = rust_data[label].get("avg_tps")
    p_str = f"{p_tps:>6.0f} tok/s" if p_tps else "       N/A"
    r_str = f"{r_tps:>6.0f} tok/s" if r_tps else "       N/A"
    ratio_p_r = f"{p_tps / r_tps:.1f}x" if (p_tps and r_tps and r_tps > 0) else "     N/A"
    ratio_r_p = f"{r_tps / p_tps:.1f}x" if (p_tps and r_tps and p_tps > 0) else "     N/A"
    print(f"{label:>8} | {p_str:>10} | {r_str:>10} | {ratio_p_r:>12} | {ratio_r_p:>12}")
