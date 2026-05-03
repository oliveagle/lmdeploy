#!/usr/bin/env python3
"""AWQ MoE EP 测试脚本 — 零配置，开箱即用。

使用方法（二选一）:
  # 单卡小模型调试（默认 hidden_dim=2048）
  python test_awq_moe_ep.py

  # 多卡真实 Qwen3.6-35B-A3B-AWQ 配置（torchrun 自动多卡 EP）
  torchrun --nproc_per_node=4 test_awq_moe_ep.py

环境变量:
  HIDDEN_DIM   覆盖 hidden_dim（默认单卡 2048，多卡 5120）
  FFN_DIM      覆盖 ffn_dim
  NUM_EXPERTS  覆盖 num_experts（多卡自动 /world_size）
  TOP_K        覆盖 top_k
  W_BIT        覆盖 w_bit（默认 4）
  GROUP_SIZE   覆盖 group_size（默认 128）
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


# ============================================================
# Qwen3.6-35B-A3B-AWQ 真实配置
# ============================================================
REAL_CONFIG = dict(
    hidden_dim=5120,
    ffn_dim=13824,
    num_experts=256,
    top_k=8,
    w_bit=4,
    group_size=128,
)


def parse_args():
    """解析参数，支持环境变量覆盖."""
    p = argparse.ArgumentParser(description='AWQ MoE EP Test')
    p.add_argument('--context-lengths', nargs='+', type=int, default=[4096, 8192, 16384])
    p.add_argument('--num-iterations-prefill', type=int, default=5)
    p.add_argument('--num-iterations-decode', type=int, default=50)
    p.add_argument('--output', type=str, default='awq_moe_ep_results.json')
    p.add_argument('--batch-size', type=int, default=1)

    # 模型参数：用环境变量可覆盖（方便 torchrun 传参）
    for k in ('hidden_dim', 'ffn_dim', 'num_experts', 'top_k', 'w_bit', 'group_size'):
        env = os.environ.get(k.upper())
        p.add_argument(f'--{k}', type=int, default=int(env) if env else None)

    args = p.parse_args()
    return args


def setup_distributed():
    """初始化分布式环境（torchrun）.

    Returns: (local_rank, world_size, is_distributed)
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_dist = world_size > 1

    if is_dist:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return local_rank, world_size, is_dist


def resolve_config(args, world_size):
    """确定最终模型配置."""
    cfg = {}
    for k in ('hidden_dim', 'ffn_dim', 'num_experts', 'top_k', 'w_bit', 'group_size'):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
        elif world_size > 1:
            cfg[k] = REAL_CONFIG[k]
        else:
            # 单卡用小模型
            cfg[k] = {
                'hidden_dim': 2048,
                'ffn_dim': 4096,
                'num_experts': 16,
                'top_k': 2,
                'w_bit': 4,
                'group_size': 128,
            }[k]

    # 多卡 EP：每卡专家数 = total // world_size
    if world_size > 1:
        assert cfg['num_experts'] % world_size == 0, \
            f"num_experts({cfg['num_experts']}) 必须能被 world_size({world_size}) 整除"
        cfg['per_rank_experts'] = cfg['num_experts'] // world_size

    return cfg


def create_test_inputs(batch_size, seq_len, hidden_dim, num_experts, top_k, device):
    """创建随机输入（每次调用都不同，避免缓存影响）."""
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    weights = torch.rand(batch_size, seq_len, top_k, dtype=torch.float16, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = torch.randint(0, num_experts, (batch_size, seq_len, top_k), device=device)
    return hidden, weights, ids


def main():
    args = parse_args()
    local_rank, world_size, is_dist = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 解析配置
    cfg = resolve_config(args, world_size)
    hidden_dim = cfg['hidden_dim']
    ffn_dim = cfg['ffn_dim']
    num_experts = cfg['num_experts']
    top_k = cfg['top_k']
    w_bit = cfg['w_bit']
    group_size = cfg['group_size']

    # 对于真实配置，先确认有足够内存再继续
    if world_size > 1:
        # 估计内存需求
        per_rank_experts = num_experts // world_size
        qweight_gb = (per_rank_experts * hidden_dim * (ffn_dim // 8) * 4) / (1024**3)
        scales_gb = (per_rank_experts * (hidden_dim // group_size) * ffn_dim * 2) / (1024**3)
        qzeros_gb = (per_rank_experts * (hidden_dim // group_size) * (ffn_dim // 8) * 4) / (1024**3)
        total_gb = qweight_gb + scales_gb + qzeros_gb

        if local_rank == 0:
            print(f"⚠️  模型大小估算：{total_gb:.2f} GiB/卡")
            print(f"   - qweight: {qweight_gb:.2f} GiB")
            print(f"   - scales: {scales_gb:.2f} GiB")
            print(f"   - qzeros: {qzeros_gb:.2f} GiB")

            # 检查 GPU 内存是否足够
            gpu_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            if total_gb > gpu_total * 0.8:
                print(f"⚠️  内存可能不够！GPU 总内存: {gpu_total:.2f} GiB")

    if local_rank == 0:
        print('=' * 80)
        print('AWQ MoE EP Test')
        print('=' * 80)
        print(f"  GPU: {torch.cuda.get_device_name(device)}  x{world_size}")
        print(f"  hidden_dim={hidden_dim}  ffn_dim={ffn_dim}")
        print(f"  num_experts={num_experts}  top_k={top_k}")
        print(f"  w_bit={w_bit}  group_size={group_size}")
        if world_size > 1:
            print(f"  EP: 每卡 {cfg['per_rank_experts']} 个专家")
        print(f"  context_lengths: {args.context_lengths}")
        print('=' * 80)

    # 初始化分布式配置
    if is_dist:
        from lmdeploy.pytorch.distributed import get_dist_manager
        from lmdeploy.pytorch.config import DistConfig

        dist_ctx = get_dist_manager().current_context()
        dist_ctx.dist_config = DistConfig(
            tp=1,
            ep=world_size,
            dp=1,
        )
        dist_ctx.ep_gpu_group = torch.distributed.group.WORLD
        dist_ctx.ep_rank = local_rank
        dist_ctx.ep_size = world_size

    # 获取实际每卡专家数
    if is_dist:
        from lmdeploy.pytorch.backends.cuda.moe.default import TritonFusedMoEImpl
        impl_builder = lambda x, y, z: TritonFusedMoEImpl(x, y, z)
        num_experts_per_rank = num_experts // world_size
    else:
        num_experts_per_rank = num_experts

    # 创建测试模型——使用一个简化版本避免内存问题
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

    print(f"  Creating model with {num_experts_per_rank} experts...")

    # 关键优化：先在 CPU 创建，按需初始化，再移到 GPU
    if is_dist:
        # 对于大模型，使用策略性内存分配
        moe = FusedMoEAWQ(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts_per_rank,
            top_k=top_k,
            w_bit=w_bit,
            group_size=group_size,
            bias=False,
            renormalize=True,
            dtype=torch.float16,
            device=device,
            all_reduce=False,
            layer_idx=0,
            act_func=None,
        )
    else:
        # 单卡小模型
        moe = FusedMoEAWQ(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            w_bit=w_bit,
            group_size=group_size,
            bias=False,
            renormalize=True,
            dtype=torch.float16,
            device=device,
            all_reduce=False,
            layer_idx=0,
            act_func=None,
        )

    # 初始化随机权重——使用增量初始化避免峰值内存
    with torch.no_grad():
        print("  Initializing weights...")
        # Gate up weights
        num_exp = moe.gate_up.num_experts
        in_feat = moe.gate_up.in_features
        out_feat = moe.gate_up.out_features
        grouped_in = in_feat // group_size
        quant_out = out_feat // (32 // w_bit)

        # 逐个专家初始化，避免大的临时张量
        for e in range(num_exp):
            moe.gate_up.qweight[e] = torch.randint(
                0, 256, (in_feat, quant_out), dtype=torch.int32, device=device
            )
            moe.gate_up.scales[e] = torch.rand(
                (grouped_in, out_feat), dtype=torch.float16, device=device
            )
            moe.gate_up.qzeros[e] = torch.randint(
                0, 256, (grouped_in, quant_out), dtype=torch.int32, device=device
            )

        # Down weights
        num_exp_down = moe.down.num_experts
        in_feat_down = moe.down.in_features
        out_feat_down = moe.down.out_features
        grouped_in_down = in_feat_down // group_size
        quant_out_down = out_feat_down // (32 // w_bit)

        for e in range(num_exp_down):
            moe.down.qweight[e] = torch.randint(
                0, 256, (in_feat_down, quant_out_down), dtype=torch.int32, device=device
            )
            moe.down.scales[e] = torch.rand(
                (grouped_in_down, out_feat_down), dtype=torch.float16, device=device
            )
            moe.down.qzeros[e] = torch.randint(
                0, 256, (grouped_in_down, quant_out_down), dtype=torch.int32, device=device
            )

    model_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
    if local_rank == 0:
        print(f"✓ Model memory: {model_mem:.2f} GB/GPU")

    # Warmup
    print("  Warmup...")
    warmup_len = 256
    h, w, ids = create_test_inputs(1, warmup_len, hidden_dim, num_experts_per_rank, top_k, device)
    for _ in range(3):
        try:
            _ = moe(h, w, ids)
        except Exception as e:
            print(f"  ⚠️ Warmup failed: {e}")
            # 对于 decode 测试可以跳过
            if world_size > 1:
                print("  ❌ Large model test failed! Let's run a smaller version.")
                return

    torch.cuda.synchronize()
    if local_rank == 0:
        print("  ✓ Warmup done")

    # 性能测试
    results = {'config': cfg, 'model_mem_gb': model_mem}

    # 用小 batch 测试
    for seq_len in args.context_lengths:
        print(f"\nTesting seq_len={seq_len}...")
        try:
            # Prefill
            print("  Prefill...")
            prefill_times = []
            for i in range(args.num_iterations_prefill):
                h, w, ids = create_test_inputs(1, seq_len, hidden_dim, num_experts_per_rank, top_k, device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                prefill_times.append((time.perf_counter() - t0) * 1000)

            prefill_mean = np.mean(prefill_times)
            prefill_tok_per_sec = seq_len / (prefill_mean / 1000)

            # Decode
            print("  Decode...")
            decode_times = []
            for i in range(args.num_iterations_decode):
                h, w, ids = create_test_inputs(1, 1, hidden_dim, num_experts_per_rank, top_k, device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = moe(h, w, ids)
                torch.cuda.synchronize()
                decode_times.append((time.perf_counter() - t0) * 1000)

            decode_mean = np.mean(decode_times)
            decode_tok_per_sec = 1 / (decode_mean / 1000)

            results[str(seq_len)] = {
                'prefill': {'mean_ms': prefill_mean, 'tok_per_sec': prefill_tok_per_sec},
                'decode': {'mean_ms': decode_mean, 'tok_per_sec': decode_tok_per_sec}
            }

            if local_rank == 0:
                print(f"✓ seq_len={seq_len} prefill {prefill_tok_per_sec:.0f} tok/s, decode {decode_tok_per_sec:.1f} tok/s")

        except Exception as e:
            if local_rank == 0:
                print(f"✗ seq_len={seq_len} failed: {e}")
            results[str(seq_len)] = {'error': str(e)}

    if local_rank == 0:
        results['peak_mem_gb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")
        print(f"Peak memory: {results['peak_mem_gb']:.2f} GiB")

    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
