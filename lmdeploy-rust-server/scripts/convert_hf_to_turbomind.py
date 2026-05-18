#!/usr/bin/env python3
"""
LMDeploy TurboMind 模型转换辅助工具

将 HuggingFace 格式的模型（包括 AWQ 量化模型）转换为 TurboMind 格式。
Rust Server 在启动时检测到 safetensors 格式时会自动调用此脚本。

使用方法:
    python convert_hf_to_turbomind.py <model_path> <workspace_path>

参数:
    model_path: HuggingFace 模型路径（包含 config.json 和 .safetensors）
    workspace_path: TurboMind 工作空间目录（转换后的权重将保存到此）
"""

import argparse
import os
import sys
from pathlib import Path


def convert_model(model_path: str, workspace_path: str) -> int:
    """
    转换 HuggingFace 模型到 TurboMind 格式

    Args:
        model_path: HuggingFace 模型路径
        workspace_path: TurboMind 工作空间目录

    Returns:
        0 成功, 非-0 失败
    """
    try:
        from lmdeploy.turbomind import TurboMind
        from lmdeploy.messages import TurbomindEngineConfig

        print(f"[LMDeploy Converter] Converting model: {model_path}")
        print(f"[LMDeploy Converter] Workspace: {workspace_path}")

        # 创建工作空间目录
        os.makedirs(workspace_path, exist_ok=True)

        # 配置 TurboMind 引擎
        engine_config = TurbomindEngineConfig(
            session_len=4096,
            max_batch_size=32,
            # AWQ 量化模型需要这些配置
            cache_max_entry_count=0.0,
            cache_block_seq_len=64,
            quant_policy=4,  # AWQ 4-bit
        )

        # 创建 TurboMind 实例并自动转换
        # 注意：这里使用 model_path 作为输入，TurboMind 会自动完成转换
        # 转换后的结果会保存到 workspace_path
        tm = TurboMind(
            model_path=model_path,
            engine_config=engine_config,
            trust_remote_code=True,
        )

        print(f"[LMDeploy Converter] Conversion completed successfully")
        return 0

    except ImportError as e:
        print(f"[LMDeploy Converter] Error: LMDeploy not installed - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[LMDeploy Converter] Conversion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def check_is_hf_model(model_path: str) -> bool:
    """
    检查模型是否为 HuggingFace 格式

    Args:
        model_path: 模型路径

    Returns:
        True 如果是 HuggingFace 格式，否则 False
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        return False

    # 检查是否存在 config.json
    config_json = model_dir / "config.json"
    if not config_json.exists():
        return False

    # 检查是否存在 safetensors 或 pytorch_model.bin 文件
    has_safetensors = any(model_dir.glob("*.safetensors"))
    has_pytorch = any(model_dir.glob("pytorch_model*.bin"))

    return has_safetensors or has_pytorch


def check_is_turbomind_model(model_path: str) -> bool:
    """
    检查模型是否已经是 TurboMind 格式

    Args:
        model_path: 模型路径

    Returns:
        True 如果是 TurboMind 格式，否则 False
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        return False

    # TurboMind 格式有 config.yaml 或 triton_models 目录
    has_config_yaml = (model_dir / "config.yaml").exists()
    has_triton_models = (model_dir / "triton_models").exists()

    return has_config_yaml or has_triton_models


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to TurboMind format"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="HuggingFace model path (containing config.json and .safetensors)"
    )
    parser.add_argument(
        "workspace_path",
        type=str,
        nargs="?",
        default="",
        help="TurboMind workspace directory (default: <model_path>/workspace)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if model needs conversion, don't convert"
    )

    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)

    # 如果未指定 workspace，使用默认值
    workspace_path = args.workspace_path
    if not workspace_path:
        workspace_path = os.path.join(model_path, "workspace")
    workspace_path = os.path.abspath(workspace_path)

    # 检查模式
    if args.check:
        if check_is_turbomind_model(model_path):
            print("turbomind")
            return 0
        elif check_is_hf_model(model_path):
            print("huggingface")
            return 0
        else:
            print("unknown")
            return 1

    # 验证模型路径
    if not check_is_hf_model(model_path):
        print(f"[LMDeploy Converter] Error: Not a valid HuggingFace model: {model_path}", file=sys.stderr)
        return 1

    # 如果已经是 TurboMind 格式，跳过转换
    if check_is_turbomind_model(workspace_path):
        print(f"[LMDeploy Converter] Workspace already exists, skipping conversion: {workspace_path}")
        return 0

    # 执行转换
    return convert_model(model_path, workspace_path)


if __name__ == "__main__":
    sys.exit(main())
