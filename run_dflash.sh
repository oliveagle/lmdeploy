#!/bin/bash
# DFlash 测试启动脚本

source /home/oliveagle/venvs/lmdeploy/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LMDEPLOY_LOG_LEVEL=INFO

echo "========================================="
echo "DFlash Speculative Decoding 测试"
echo "========================================="
echo ""

cd /home/oliveagle/opt/lmdeploy/lmdeploy

time python test_dflash.py --tp 4 --num-spec-tokens 8 --max-tokens 64 --prompt "你好，请简单介绍一下你自己。"
