#!/usr/bin/env python3
"""快速环境检查 - 验证能否运行基线测试"""

import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LMDEPLOY_EXECUTOR_BACKEND'] = 'mp'

print("=" * 70)
print("环境检查")
print("=" * 70)

# 1. 检查 CUDA
print("\n1. CUDA 检查")
try:
    import torch
    print(f"   ✅ PyTorch 版本: {torch.__version__}")
    print(f"   ✅ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA 版本: {torch.version.cuda}")
        print(f"   ✅ GPU 数量: {torch.cuda.device_count()}")
        for i in range(min(4, torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(i)
            print(f"   ✅ GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
except Exception as e:
    print(f"   ❌ CUDA 检查失败: {e}")
    sys.exit(1)

# 2. 检查 LMDeploy
print("\n2. LMDeploy 检查")
try:
    import lmdeploy
    print(f"   ✅ LMDeploy 版本: {lmdeploy.__version__}")
except Exception as e:
    print(f"   ❌ LMDeploy 检查失败: {e}")
    sys.exit(1)

# 3. 检查配置类
print("\n3. 配置类检查")
try:
    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.messages import QuantPolicy
    print("   ✅ pipeline")
    print("   ✅ PytorchEngineConfig")
    print(f"   ✅ QuantPolicy (TURBO_QUANT={QuantPolicy.TURBO_QUANT})")

    # 创建配置
    config = PytorchEngineConfig(
        tp=1,
        ep=4,
        attn_tp_size=1,
        moe_tp_size=1,
        session_len=1024,
        quant_policy=QuantPolicy.TURBO_QUANT,
    )
    print(f"   ✅ 配置创建成功: ep={config.ep}, tp={config.tp}, quant_policy={config.quant_policy}")
except Exception as e:
    print(f"   ❌ 配置类检查失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 检查模型路径
print("\n4. 模型检查")
model_path = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3___6-35B-A3B-AWQ"
if Path(model_path).exists():
    print(f"   ✅ 模型路径存在: {model_path}")
    # 检查关键文件
    config_file = Path(model_path) / "config.json"
    if config_file.exists():
        print(f"   ✅ config.json 存在")
    else:
        print(f"   ⚠️  config.json 不存在")
else:
    print(f"   ❌ 模型路径不存在: {model_path}")
    print("   请确保模型已下载到正确位置")

# 5. 显存检查
print("\n5. 显存检查")
try:
    import torch
    if torch.cuda.is_available():
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(4)) / 1024**3
        print(f"   ✅ 4x GPU 总显存: {total_memory:.1f} GB")
        if total_memory >= 64:
            print(f"   ✅ 显存充足 (需要 ~60 GB)")
        else:
            print(f"   ⚠️  显存可能不足 (需要 ~60 GB)")
except Exception as e:
    print(f"   ❌ 显存检查失败: {e}")

print("\n" + "=" * 70)
print("✅ 环境检查完成 - 可以运行基线测试")
print("=" * 70)
print("\n运行基线测试:")
print("  ./test_baseline_ep4_tp1_turboquant.sh")
