#!/usr/bin/env python3
"""
临时补丁：强制使用 gloo backend 而不是 nccl
用于测试不支持 P2P 的 GPU 组合
"""
import os
import sys


def main():
    # 强制使用 gloo backend
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # 修改 lmdeploy 的 distributed 模块
    import lmdeploy.pytorch.distributed as dist_module

    # 保存原始的 build 方法
    original_build = dist_module.DistContext.build

    @classmethod
    def patched_build(cls, rank=0, dist_config=None, ccl_backend='gloo'):  # 改为 gloo
        """强制使用 gloo backend."""
        return original_build(rank=rank, dist_config=dist_config, ccl_backend=ccl_backend)

    # 应用补丁
    dist_module.DistContext.build = patched_build

    print("✓ 已应用补丁：强制使用 gloo backend")

    # 现在运行测试
    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

    print("\n加载模型 (TP=4, TurboQuant)...")
    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.8,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        quant_policy=QuantPolicy.TURBO_QUANT,
        dtype='float16',
    )

    try:
        import time
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        load_time = time.time() - start
        print(f"✓ 模型加载成功 ({load_time:.2f}s)")

        print("\n测试推理...")
        response = pipe("你好，请介绍一下你自己。", max_new_tokens=128, do_sample=False)
        print(f"输出: {response.text}")
        print("\n✓ 测试成功！")
        return 0
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
