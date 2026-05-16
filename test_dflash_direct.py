#!/usr/bin/env python3
"""
DFlash 直接测试 - 使用 TurboMind API
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'INFO'

from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig, SpeculativeConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"


def main():
    print("=" * 60)
    print("DFlash 直接测试 - TurboMind API")
    print("=" * 60)

    # Create speculative config
    speculative_config = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
        quant_policy=0,
    )

    # Create engine config
    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tp=1,
        cache_max_entry_count=0.4,
        quant_policy=8,
        session_len=4096,
    )
    tm_config.speculative_config = speculative_config

    print("创建 TurboMind...")
    try:
        tm = TurboMind.from_pretrained(
            target_model,
            engine_config=tm_config,
        )
        print("✓ TurboMind 创建成功！\n")
    except Exception as e:
        print(f"✗ TurboMind 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get instance
    print("获取 TurboMindInstance...")
    try:
        instance = tm.model_inst
        print(f"✓ TurboMindInstance: {instance}\n")
    except Exception as e:
        print(f"✗ 获取实例失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create instance
    print("创建实例...")
    try:
        instance = tm.create_instance(0)
        print(f"✓ 实例已创建\n")
    except Exception as e:
        print(f"✗ 创建实例失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Done!")


if __name__ == "__main__":
    main()