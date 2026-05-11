#!/usr/bin/env python3
"""
V100 + AWQ MoE 测试 (仅 V100, 不含 PG199)
"""
import os
import sys
import time

# V100 组 (GPU 0-3), PG199 不包含在内
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# NCCL 配置 - V100 NVLink 可能没全部连接, 禁用 P2P 避免报错
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_IB_DISABLE'] = '1'

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"


def print_gpu_info():
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        gpus = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        print(f"  CUDA_VISIBLE_DEVICES=0,1,2,3 -> {len(gpus)} 块 GPU 可见:")
        for i, gpu in enumerate(gpus[:4]):
            parts = [p.strip() for p in gpu.split(',')]
            if len(parts) == 4:
                print(f"    GPU {parts[0]}: {parts[1]} - {parts[2]} MB / {parts[3]} MB")
    except Exception:
        pass


def test_v100_awq():
    print("=" * 70)
    print("V100 分组: AWQ MoE 测试 (TP=4)")
    print("=" * 70)
    print(f"模型: {MODEL_PATH}")

    print_gpu_info()

    from lmdeploy import pipeline, PytorchEngineConfig

    engine_config = PytorchEngineConfig(
        tp=4,
        session_len=2048,
        cache_max_entry_count=0.6,
        max_batch_size=1,
        block_size=64,
        eager_mode=True,
        dtype='float16',
    )

    try:
        print("\n加载模型...")
        start = time.time()
        pipe = pipeline(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            backend_config=engine_config,
        )
        print(f"✓ 模型加载成功 ({time.time()-start:.2f}s)")

        print_gpu_info()

        prompts = ["你好", "What is AI?"]
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 测试 {i}: {prompt} ---")
            start = time.time()
            response = pipe(prompt, max_new_tokens=32, do_sample=False)
            elapsed = time.time() - start
            print(f"输出: {response.text[:100]}")
            print(f"生成: {response.generate_token_len} tokens, {elapsed:.2f}s, "
                  f"{response.generate_token_len/max(elapsed,0.001):.1f} tok/s")

        print("\n✓ 全部测试成功!")
        return True
    except Exception as e:
        import traceback
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("V100 + AWQ MoE 修复验证")
    print("=" * 70)
    print("GPU: 4x V100 (独立分组, 不含 PG199)")
    print(f"修复: topk_ids 验证 / clamp 防止非法 expert ID")
    print()

    success = test_v100_awq()

    print("\n" + "=" * 70)
    print(f"{'✓ 成功' if success else '✗ 失败'}")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
