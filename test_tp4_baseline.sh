#!/bin/bash
# Test 1: TP=4, NO EP, NO TurboQuant (baseline that should work)

set -e

cat > /tmp/test_tp4_nodflash.py << 'PYEOF'
#!/usr/bin/env python3
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
sys.path.insert(0, '/home/oliveagle/opt/lmdeploy/lmdeploy')

import time
from lmdeploy import pipeline, PytorchEngineConfig

print("=" * 70)
print("Test: TP=4, NO EP, NO TurboQuant")
print("=" * 70)

pipe = pipeline(
    model_path="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ",
    backend_config=PytorchEngineConfig(
        tp=4,
        ep=1,
        session_len=512,
        cache_max_entry_count=0.7,
        max_batch_size=2,
        block_size=64,
        eager_mode=True,
    ),
)

print("推理...")
t0 = time.time()
response = pipe("你好，请简短回答。", max_new_tokens=64, do_sample=False)
t1 = time.time()
output = response.text if hasattr(response, 'text') else str(response)
print(f"输出: {output[:200]}")
print(f"时间: {t1-t0:.1f}秒")
print("=" * 70)
print("✅ 完成")
print("=" * 70)
PYEOF

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_tp4_nodflash.py