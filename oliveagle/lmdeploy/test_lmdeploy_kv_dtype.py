#!/usr/bin/env python3
"""测试 LMDeploy KV cache 配置"""
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
import time

MODEL = '/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ'

# 测试不同的 kv_cache_dtype
kv_dtypes = ['fp16', 'fp8_e5m2', 'fp8_e4m3', 'int8']

for kv_dtype in kv_dtypes:
    print(f"\n{'='*60}")
    print(f"测试 kv_cache_dtype: {kv_dtype}")
    print(f"{'='*60}")

    for session_len in [4096, 8192, 16384, 32768]:
        print(f"\n  session_len={session_len}...", end=' ', flush=True)
        try:
            backend_config = TurbomindEngineConfig(
                session_len=session_len,
                cache_max_entry_count=0.95,
                kv_cache_dtype=kv_dtype,
            )
            pipe = pipeline(
                model_path=MODEL,
                backend_config=backend_config,
                tp=1,
                log_level='ERROR',
            )

            # 测试推理
            response = pipe.generate([100]*32, GenerationConfig(max_new_tokens=8, temperature=0))
            print(f"✓ OK")
            pipe.close()
        except Exception as e:
            err = str(e)[:100]
            print(f"✗ {err}")
            if 'truncated' in err.lower():
                print(f"     (session_len 被截断)")
