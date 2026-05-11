#!/usr/bin/env python3
"""
测试 LMDeploy + DFlash + TurboQuant
分阶段测试:
1. 基础 LMDeploy + TurboQuant
2. LMDeploy + DFlash + TurboQuant
"""
import os
import sys
import time

# GPU 配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_DEBUG'] = 'INFO'

# 路径
MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
DRAFT_MODEL = "/home/oliveagle/.cache/modelscope/hub/models/z-lab/Qwen3___6-35B-A3B-DFlash"


def print_gpu_memory():
    """打印 GPU 显存使用"""
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True)
        print("GPU 显存:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3:
                    print(f"  GPU {parts[0].strip()}: {parts[1].strip()}MB 已用 / {parts[2].strip()}MB 空闲")
    except Exception:
        pass


def test_pytorch_turboquant():
    """测试 1: PyTorch Backend + TurboQuant"""
    print("\n" + "=" * 70)
    print("测试 1: PyTorch Backend + TurboQuant (TP=4)")
    print("=" * 70)

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=4096,
        cache_max_entry_count=0.8,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,  # 禁用 CUDA graph
        quant_policy=QuantPolicy.TURBO_QUANT,
        dtype='float16',
    )

    try:
        print("加载模型...")
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        load_time = time.time() - start
        print(f"✓ 模型加载成功 ({load_time:.2f}s)")

        print_gpu_memory()

        print("\n测试推理...")
        prompts = [
            "你好，请介绍一下你自己。",
            "What is machine learning?",
            "Python 中如何实现多线程?",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 测试 {i}: {prompt[:30]}... ---")
            gen_start = time.time()
            response = pipe(
                prompt,
                max_new_tokens=64,
                do_sample=False,
            )
            gen_time = time.time() - gen_start

            print(f"输出: {response.text[:100]}")
            print(f"生成: {response.generate_token_len} tokens, {gen_time:.2f}s, {response.generate_token_len/max(gen_time,0.01):.1f} tok/s")

        print("\n✓ 测试 1 成功!")
        return True

    except Exception as e:
        import traceback
        print(f"✗ 测试 1 失败: {e}")
        traceback.print_exc()
        return False


def test_pytorch_dflash():
    """测试 2: PyTorch Backend + DFlash + TurboQuant"""
    print("\n" + "=" * 70)
    print("测试 2: PyTorch Backend + DFlash + TurboQuant (TP=4)")
    print("=" * 70)

    # 检查 DFlash 集成状态
    try:
        from lmdeploy.pytorch.spec_decode import DFlashSpecAgent
        print("✓ DFlash 模块已安装")
    except ImportError as e:
        print(f"✗ DFlash 模块未安装: {e}")
        print("  跳过 DFlash 测试")
        return False

    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy

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

    # DFlash 配置
    speculative_config = dict(
        backend='dflash',
        draft_model_path=DRAFT_MODEL,
        num_spec_tokens=4,  # 猜测 token 数
    )

    try:
        print("加载主模型 + DFlash draft 模型...")
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=engine_config,
            speculative_config=speculative_config,
        )
        load_time = time.time() - start
        print(f"✓ 模型加载成功 ({load_time:.2f}s)")

        print_gpu_memory()

        print("\n测试推理 (DFlash 加速)...")
        prompts = [
            "请解释 Transformer 架构。",
            "How does attention mechanism work?",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 测试 {i}: {prompt[:30]}... ---")
            gen_start = time.time()
            response = pipe(
                prompt,
                max_new_tokens=64,
                do_sample=False,
            )
            gen_time = time.time() - gen_start

            print(f"输出: {response.text[:100]}")
            print(f"生成: {response.generate_token_len} tokens, {gen_time:.2f}s")

            # DFlash 统计
            if hasattr(response, 'dflash_stats'):
                stats = response.dflash_stats
                print(f"  DFlash: 接受率={stats.get('accept_rate', 0):.2%}, "
                      f"猜测={stats.get('num_speculated', 0)}, "
                      f"接受={stats.get('num_accepted', 0)}")

        print("\n✓ 测试 2 成功!")
        return True

    except Exception as e:
        import traceback
        print(f"✗ 测试 2 失败: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("LMDeploy + DFlash + TurboQuant 测试")
    print("=" * 70)
    print(f"主模型: {MODEL_PATH}")
    print(f"Draft 模型: {DRAFT_MODEL}")
    print(f"GPU: CUDA_VISIBLE_DEVICES=0,1,2,3")
    print()

    results = {}
    results['pytorch_turboquant'] = test_pytorch_turboquant()
    results['pytorch_dflash'] = test_pytorch_dflash()

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    return 0 if any(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
