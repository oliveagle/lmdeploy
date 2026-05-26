#!/usr/bin/env python3
"""
使用 GPU tokenizer 的 Rust server prefill benchmark
验证零拷贝路径性能提升
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count):
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats

test_cases = [("1K", 1024), ("4K", 4096), ("8K", 8192)]

# 先启动 server
import requests

results = {}

for label, target_tokens in test_cases:
    prompt = gen_prompt(target_tokens)
    actual_tokens = len(prompt.split()) * 10  # 近似

    times = []
    for r in range(3):
        start = time.perf_counter()
        resp = requests.post("http://localhost:3000/v1/completions", json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0.7,
        })
        elapsed = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            times.append(elapsed)
            print(f"{label} Run {r}: {elapsed:.1f}ms")

    if times:
        avg_ms = sum(times) / len(times)
        results[label] = {"avg_ms": avg_ms, "raw": times}

print("\nRust Server Prefill Benchmark")
for label in ["1K", "4K", "8K"]:
    if label in results:
        print(f"  {label}: {results[label]['avg_ms']:.1f}ms")
