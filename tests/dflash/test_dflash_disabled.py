#!/usr/bin/env python3
"""
测试禁用 DFlash 加载是否能解决内存分配问题
"""

import os
import sys

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

# 禁用 DFlash (使用传统的 speculative decoding)
speculative_config = SpeculativeConfig(
    method='legacy',  # 使用传统方法而非 DFlash
    model=draft_model,
    num_speculative_tokens=8,
)

tm_config = TurbomindEngineConfig(
    model_format='awq',
    tensor_parallel=1,
    cache_max_entry_count=0.2,
    quant_policy=8,
    session_len=16384,
)

gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)

def main():
    print("=" * 60)
    print("测试禁用 DFlash 加载")
    print("=" * 60)
    print(f"目标: {target_model}")
    print(f"草稿: {draft_model}")
    print(f"方法: legacy (非 DFlash)\n")

    try:
        print("创建 Pipeline...")
        pipe = pipeline(
            target_model,
            backend_config=tm_config,
            speculative_config=speculative_config,
            log_level='INFO'
        )
        print("✓ 创建成功！\n")

        prompt = "Python 是什么？"
        print(f"测试提示: {prompt}\n")

        resp = pipe(
            [{"role": "user", "content": prompt}],
            gen_config=gen_config,
            sequence_start=True,
            sequence_end=True,
            chat_template_kwargs={'enable_thinking': False}
        )

        print("\n回复:")
        print(resp.text)
        print("\n✓ 测试成功！")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
