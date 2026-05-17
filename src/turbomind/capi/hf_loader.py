#!/usr/bin/env python3
"""
Python bridge for loading HuggingFace models into TurboMind C API.
Called from C++ via popen.
"""

import sys
import os
import argparse
import json

sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/lmdeploy')

def load_hf_model(model_dir: str, device_id: int = 0, session_len: int = 8192):
    """Load HF model and initialize TurboMind."""
    from lmdeploy.turbomind.turbomind import TurboMind
    from lmdeploy.messages import TurbomindEngineConfig

    config = TurbomindEngineConfig(
        session_len=session_len,
        max_batch_size=128,
        cache_max_entry_count=0.8,
    )

    print(f"Loading model from: {model_dir}", file=sys.stderr)
    tm = TurboMind(model_dir, engine_config=config, trust_remote_code=False)

    info = {
        "vocab_size": tm._vocab_size,
        "gpu_count": tm.gpu_count,
        "devices": tm.devices,
        "session_len": tm.session_len,
        "status": "loaded"
    }

    # 保持进程运行以保持模型在内存中
    print(json.dumps(info), flush=True)

    # 等待信号退出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...", file=sys.stderr)
        tm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load HF model into TurboMind')
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--session-len', type=int, default=8192)
    args = parser.parse_args()

    load_hf_model(args.model_dir, args.device_id, args.session_len)
