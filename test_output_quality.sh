#!/bin/bash
# Test model output quality

set -e

cat > /tmp/test_output_quality.py << 'PYEOF'
#!/usr/bin/env python3
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '/home/oliveagle/opt/lmdeploy/lmdeploy')

from lmdeploy import pipeline, PytorchEngineConfig

MODEL = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

print("加载模型...")
pipe = pipeline(
    model_path=MODEL,
    backend_config=PytorchEngineConfig(
        tp=1,
        session_len=1024,
        cache_max_entry_count=0.6,
        max_batch_size=2,
        block_size=64,
        eager_mode=True,
    ),
)

# Test greedy
print("\nTest 1: greedy (do_sample=False)")
r = pipe("你好", max_new_tokens=32, do_sample=False)
print(f"  {repr(r.text[:100] if hasattr(r, 'text') else str(r))}")

# Test with sampling
print("\nTest 2: sampling (do_sample=True, temp=0.7)")
r = pipe("你好", max_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.9)
print(f"  {repr(r.text[:100] if hasattr(r, 'text') else str(r))}")

# Test with longer prompt
print("\nTest 3: longer prompt (greedy)")
r = pipe("请介绍一下Python编程语言，不少于50个字", max_new_tokens=64, do_sample=False)
print(f"  {repr(r.text[:100] if hasattr(r, 'text') else str(r))}")

print("\n完成")
PYEOF

source ~/venvs/lmdeploy/bin/activate
python3 /tmp/test_output_quality.py