#!/bin/bash
# 文件名: test_turboquant_ep4_basic.sh
#
# 简化版本: 只测试 EP=4 + TurboQuant KV cache 量化
# 不使用 DFlash，先确认基础功能能运行

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

cat > /tmp/test_turboquant_ep4_basic.py << 'EOF'
"""测试 EP=4 + TurboQuant KV cache 量化"""

import os
import sys
from pathlib import Path

LMDEPLOY_PATH = Path(__file__).parent
sys.path.insert(0, str(LMDEPLOY_PATH))

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import torch.distributed as dist

def run_worker(rank, world_size):
    """每个 rank 的执行函数"""

    # 设置分布式环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    # 初始化分布式
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        from lmdeploy.pytorch.config import ModelConfig, CacheConfig, BackendConfig
        from lmdeploy.pytorch.config import DistConfig
        from lmdeploy.pytorch.engine import Engine
        from lmdeploy.pytorch.messages import GenerationConfig
        from lmdeploy.pytorch.config import QuantPolicy

        # 基础配置
        model_path = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"

        print(f"Rank {rank}: Starting with world_size={world_size}")

        # 配置 TurboQuant KV cache
        cache_config = CacheConfig(
            quant_policy=QuantPolicy.TURBO_QUANT,  # K=4bit, V=2bit
            cache_max_entry_count=0.05,            # 极小的 KV cache
            block_size=16,
            max_prefill_iter=1
        )

        engine_config = DistConfig(
            tp=2, ep=4, moe_tp=1, attn_tp=2,
            world_size=8,
            dp=1
        )

        print(f"Rank {rank}: engine_config={engine_config}")

        # 测试初始化
        if rank == 0:
            print("Rank 0: Initializing engine...")

            engine = Engine(
                model_path,
                cache_config=cache_config
            )

            print("Rank 0: Engine created successfully!")

            # 简单测试
            prompts = [
                "How are you?",
            ]

            generation_config = GenerationConfig(
                max_new_tokens=32,
                temperature=0.0
            )

            for i, out in enumerate(engine.generate(prompts, generation_config)):
                print(f"Rank 0: Output {i}: {out.response}")

        else:
            # 其他 ranks 等待指令
            print(f"Rank {rank}: Waiting for work...")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n=== ERROR in Rank {rank} ===")
        print(f"Type: {type(e)}")
        print(f"Message: {e}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    """主函数 - 使用 8 GPUs (TP=2, EP=4)"""

    print("=== EP4 + TurboQuant 测试 ===")
    print("Model: Qwen3___6-35B-A3B-AWQ")
    print("Config: TP=2, EP=4, TurboQuant KV cache")

    world_size = 8
    processes = []

    import torch.multiprocessing as mp
    for i in range(world_size):
        p = mp.Process(target=run_worker, args=(i, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("=== Done ===")

if __name__ == "__main__":
    main()
EOF

echo "Running TurboQuant EP4 test..."
python /tmp/test_turboquant_ep4_basic.py
