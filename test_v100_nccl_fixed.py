#!/usr/bin/env python3
"""
V100 + AWQ MoE 测试 - 强化 NCCL 配置
"""
import os
import sys
import time

# ===== GPU 分组: 仅 V100 (0-3), 排除 PG199 (4) =====
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# ===== NCCL 配置 - 避免 P2P 问题 =====
# 禁用 P2P, 强制使用 Ring 算法 (不依赖 P2P)
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_ALGO'] = 'Ring'
os.environ['NCCL_PROTO'] = 'Simple'
# 禁用 InfiniBand (本地机器不需要)
os.environ['NCCL_IB_DISABLE'] = '1'
# 使用 SHM 进行同节点通信
os.environ['NCCL_SHM_DISABLE'] = '0'
# 减少日志噪音
os.environ['NCCL_DEBUG'] = 'WARN'

MODEL_PATH = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"


def print_gpu_info():
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    gpus = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
    v100_count = sum(1 for g in gpus if 'V100' in g)
    print(f"  可见 GPU: {len(gpus)} 块 (V100: {v100_count} 块)")
    for g in gpus:
        parts = [p.strip() for p in g.split(',')]
        if len(parts) == 4:
            print(f"    GPU {parts[0]}: {parts[1]} - {parts[2]} MB / {parts[3]} MB")


def main():
    print("=" * 70)
    print("V100 + AWQ MoE 修复验证")
    print("=" * 70)
    print(f"模型: {MODEL_PATH}")
    print(f"NCCL_ALGO={os.environ['NCCL_ALGO']}  NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']}")
    print()

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
        print()

        print_gpu_info()

        prompts = ["你好", "What is AI?"]
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 测试 {i}: {prompt} ---")
            start = time.time()
            response = pipe(prompt, max_new_tokens=32, do_sample=False)
            elapsed = time.time() - start
            print(f"输出: {response.text[:100]}")
            print(f"生成: {response.generate_token_len} tokens, "
                  f"{elapsed:.2f}s ({response.generate_token_len/max(elapsed,0.001):.1f} tok/s)")

        print("\n✓ 全部测试成功!")
        return 0
    except Exception as e:
        import traceback
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
