#!/usr/bin/env python3
"""
测试 NCCL P2P 是否工作
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def test_rank(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    print(f"Rank {rank}: Initializing NCCL...")
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    print(f"Rank {rank}: Device = {device}, Name = {torch.cuda.get_device_name(0)}")

    # 测试 all_reduce
    tensor = torch.ones(4, 4) * rank
    tensor = tensor.to(device)
    print(f"Rank {rank}: Before all_reduce, tensor[0,0] = {tensor[0,0].item()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: After all_reduce, tensor[0,0] = {tensor[0,0].item()} (expected = {sum(range(world_size))})")

    dist.destroy_process_group()
    print(f"Rank {rank}: Done")


if __name__ == '__main__':
    world_size = 4
    mp.spawn(test_rank, args=(world_size,), nprocs=world_size, join=True)
