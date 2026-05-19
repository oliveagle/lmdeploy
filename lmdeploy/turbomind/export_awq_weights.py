#!/usr/bin/env python3
"""AWQ Offline Weight Converter for TurboMind.

This script converts HuggingFace AWQ safetensors to TurboMind .bin format
for offline loading by the C++ InitFromPath() API.

The conversion process:
1. Loads the AWQ model via Python TurboMind API
2. Extracts transformed weights from C++ modules (after AWQ dequantization)
3. Writes weights to model_path/workspace/tm_weights/ directory

Usage:
    python lmdeploy/turbomind/export_awq_weights.py \\
        --model-path /path/to/awq/model \\
        --output-dir /path/to/workspace/tm_weights
"""

import argparse
import json
import os
import os.path as osp
import sys
from collections import OrderedDict

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert AWQ safetensors to TurboMind .bin format"
    )
    p.add_argument("--model-path", required=True,
                   help="Path to HuggingFace AWQ model directory")
    p.add_argument("--output-dir", required=True,
                   help="Output workspace directory for .bin files")
    p.add_argument("--session-len", type=int, default=2048,
                   help="Max context length")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallelism size")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Trust remote code")
    return p.parse_args()


def get_model_info(model_path: str) -> dict:
    """Read config.json and extract model parameters."""
    config_path = osp.join(model_path, "config.json")
    if not osp.exists(config_path):
        raise RuntimeError(f"config.json not found at {model_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    # Handle nested config (text_config, model_config)
    config_source = cfg
    if "text_config" in cfg:
        config_source = cfg["text_config"]
    elif "model_config" in cfg:
        config_source = cfg["model_config"]

    # Extract model parameters
    info = {
        "hidden_size": config_source.get("hidden_size", cfg.get("hidden_size", 4096)),
        "num_layers": config_source.get("num_hidden_layers", cfg.get("num_hidden_layers", 32)),
        "num_heads": config_source.get("num_attention_heads", cfg.get("num_attention_heads", 32)),
        "num_kv_heads": config_source.get("num_key_value_heads", cfg.get("num_key_value_heads", None)),
        "vocab_size": config_source.get("vocab_size", cfg.get("vocab_size", 32000)),
        "intermediate_size": config_source.get("intermediate_size", cfg.get("intermediate_size", None)),
    }

    # Check for AWQ quantization
    quant_config = cfg.get("quantization_config", {})
    info["is_awq"] = quant_config.get("quant_method") == "awq"
    info["group_size"] = quant_config.get("group_size", 128) if info["is_awq"] else None

    return info


def export_weights_via_turbomind(
    model_path: str,
    output_dir: str,
    session_len: int,
    tp: int,
    trust_remote_code: bool
):
    """Use Python TurboMind API to build C++ model tree and dump weights."""
    # Add project root to path
    project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    sys.path.insert(0, project_root)

    import lmdeploy
    lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
    sys.path.insert(0, osp.join(lmdeploy_dir, 'lib'))

    from lmdeploy.messages import TurbomindEngineConfig  # noqa: E402
    from lmdeploy.turbomind.turbomind import TurboMind  # noqa: E402

    print(f"[AWQ Converter] Loading model from {model_path}")
    print(f"[AWQ Converter] Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Get model info
    model_info = get_model_info(model_path)
    print(f"[AWQ Converter] Model info: {json.dumps(model_info, indent=2)}")

    engine_config = TurbomindEngineConfig(
        session_len=session_len,
        max_batch_size=1,  # Use minimal batch size for conversion
        cache_block_seq_len=64,
        cache_max_entry_count=0.0,  # Don't allocate KV cache
        tp=tp,
        empty_init=False,
    )

    # Load model via TurboMind Python API
    # This builds the C++ model tree and loads weights
    tm_model = TurboMind(
        model_path,
        engine_config=engine_config,
        trust_remote_code=trust_remote_code,
    )

    print("[AWQ Converter] Model loaded successfully. Extracting weights...")

    # Extract weights from the C++ modules
    _extract_weights_from_model(tm_model, output_dir, engine_config)

    print(f"[AWQ Converter] Weights exported to {output_dir}")


def _extract_weights_from_model(tm_model, output_dir: str, engine_config):
    """Extract weights from the TurboMind model instance to disk."""

    # Get the model_comm and root module
    model_comm = tm_model.model_comm
    device_id = 0  # For single-GPU conversion

    # Get the root module (ModelRoot)
    root = model_comm.root(device_id)

    # Get the text_model child (ModelWeight)
    text_model = root.child("text_model")

    # Create weight map to store all tensors
    weight_map = OrderedDict()

    # Helper function to recursively extract weights from modules
    def extract_module_weights(module, prefix=""):
        """Recursively extract all weights from a module and its children."""
        # Extract parameters from this module
        # Note: The C++ Module API doesn't expose param names directly
        # We need to use the known structure based on module type

        module_type = module.type()
        print(f"[AWQ Converter] Processing module: {prefix} (type: {module_type})")

        # Try to get common parameters based on module type
        if module_type == "ModelWeight":
            # tok_embeddings parameter
            try:
                tok_emb_param = module.param("tok_embeddings")
                if tok_emb_param:
                    tensor = tok_emb_param.get()
                    weight_map[f"{prefix}.tok_embeddings"] = tensor
                    print(f"  [AWQ Converter]   tok_embeddings: {tensor.shape} {tensor.dtype}")
            except Exception as e:
                print(f"  [AWQ Converter]   No tok_embeddings: {e}")

            # Try children
            _try_get_child(module, "norm", prefix, weight_map)
            _try_get_child(module, "output", prefix, weight_map)
            _try_get_child(module, "layers", prefix, weight_map)

        elif module_type == "DecoderLayerWeight":
            # Try common children
            _try_get_child(module, "attention_norm", prefix, weight_map)
            _try_get_child(module, "ffn_norm", prefix, weight_map)
            _try_get_child(module, "attention", prefix, weight_map)
            _try_get_child(module, "feed_forward", prefix, weight_map)
            _try_get_child(module, "linear_attn", prefix, weight_map)
            _try_get_child(module, "moe_ffn", prefix, weight_map)

        elif module_type == "AttentionWeight":
            # Attention sub-modules
            _try_get_child(module, "w_qkv", prefix, weight_map)
            _try_get_child(module, "wo", prefix, weight_map)
            _try_get_child(module, "q_proj", prefix, weight_map)
            _try_get_child(module, "k_proj", prefix, weight_map)
            _try_get_child(module, "v_proj", prefix, weight_map)

        elif module_type == "FfnWeight":
            # FFN sub-modules
            _try_get_child(module, "w1", prefix, weight_map)
            _try_get_child(module, "w2", prefix, weight_map)
            _try_get_child(module, "w3", prefix, weight_map)

        elif module_type == "LinearWeight":
            # Extract weight parameter
            try:
                weight_param = module.param("weight")
                if weight_param:
                    tensor = weight_param.get()
                    weight_map[f"{prefix}.weight"] = tensor
                    print(f"  [AWQ Converter]   weight: {tensor.shape} {tensor.dtype}")
            except Exception as e:
                print(f"  [AWQ Converter]   No weight: {e}")

            # Try scales and zeros (for AWQ)
            try:
                scales_param = module.param("scales")
                if scales_param:
                    tensor = scales_param.get()
                    weight_map[f"{prefix}.scales"] = tensor
                    print(f"  [AWQ Converter]   scales: {tensor.shape} {tensor.dtype}")
            except Exception:
                pass

            try:
                zeros_param = module.param("zeros")
                if zeros_param:
                    tensor = zeros_param.get()
                    weight_map[f"{prefix}.zeros"] = tensor
                    print(f"  [AWQ Converter]   zeros: {tensor.shape} {tensor.dtype}")
            except Exception:
                pass

        elif module_type == "NormWeight":
            # Extract weight parameter
            try:
                weight_param = module.param("weight")
                if weight_param:
                    tensor = weight_param.get()
                    weight_map[f"{prefix}.weight"] = tensor
                    print(f"  [AWQ Converter]   weight: {tensor.shape} {tensor.dtype}")
            except Exception as e:
                print(f"  [AWQ Converter]   No weight: {e}")

        elif module_type == "ModuleList":
            # Iterate through list items
            idx = 0
            while True:
                try:
                    child = module.child(str(idx))
                    if child is None:
                        break
                    extract_module_weights(child, f"{prefix}.{idx}")
                    idx += 1
                except Exception:
                    break

    def _try_get_child(module, name, prefix, weight_map):
        """Try to get a child and extract its weights."""
        try:
            child = module.child(name)
            if child:
                extract_module_weights(child, f"{prefix}.{name}")
        except Exception as e:
            print(f"  [AWQ Converter]   No child {name}: {e}")

    # Start extraction from text_model
    extract_module_weights(text_model, "text_model")

    print(f"[AWQ Converter] Extracted {len(weight_map)} tensors")

    # Write weights to disk
    _write_weights_to_disk(weight_map, output_dir)

    # Write config.yaml
    _write_config_yaml(tm_model, output_dir, engine_config)


def _write_weights_to_disk(weight_map: OrderedDict, output_dir: str):
    """Write extracted weights to .bin files."""
    print(f"[AWQ Converter] Writing {len(weight_map)} tensors to disk...")

    # Group tensors by module for efficient file organization
    # For simplicity, write all tensors to a single file with metadata
    index_path = osp.join(output_dir, "weight_index.json")
    bin_path = osp.join(output_dir, "weights.bin")

    # Build index
    index = {
        "metadata": {
            "total_tensors": len(weight_map),
            "format": "turbomind_awq_v1",
        },
        "weight_map": {},
    }

    # Write tensors to binary file
    offset = 0
    with open(bin_path, "wb") as f:
        for name, tensor in weight_map.items():
            # Get tensor data via dlpack
            dlpack_capsule = tensor.__dlpack__()
            import torch
            torch_tensor = torch.from_dlpack(dlpack_capsule)

            # Align to 64-byte boundary
            aligned_offset = (offset + 63) & ~63
            if aligned_offset > offset:
                padding = aligned_offset - offset
                f.write(b"\x00" * padding)
                offset = aligned_offset

            # Write tensor info
            tensor_size = torch_tensor.numel() * torch_tensor.element_size()

            index["weight_map"][name] = {
                "offset": offset,
                "dtype": str(torch_tensor.dtype),
                "shape": list(torch_tensor.shape),
                "size": tensor_size,
            }

            # Write tensor data
            if torch_tensor.dtype == torch.bfloat16:
                data = torch_tensor.float().numpy().view(np.uint16).tobytes()
            elif torch_tensor.dtype == torch.float16:
                data = torch_tensor.numpy().view(np.uint16).tobytes()
            elif torch_tensor.dtype == torch.float32:
                data = torch_tensor.numpy().view(np.uint32).tobytes()
            else:
                data = torch_tensor.numpy().tobytes()

            f.write(data)
            offset += tensor_size

    # Write index
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[AWQ Converter] Wrote {bin_path} ({offset} bytes)")
    print(f"[AWQ Converter] Wrote {index_path}")


def _write_config_yaml(tm_model, output_dir: str, engine_config):
    """Write the TurboMind config.yaml."""
    import _turbomind as _tm  # noqa: F401

    config_path = osp.join(output_dir, "config.yaml")

    # Extract model info from the text model
    text_model = tm_model.text_model if hasattr(tm_model, 'text_model') else None

    if text_model:
        cfg = text_model.cfg
        model_config = {
            "model_name": getattr(tm_model, 'model_name', 'exported_model'),
            "tensor_para_size": engine_config.tp,
            "head_num": getattr(cfg, 'num_attention_heads', 32),
            "kv_head_num": getattr(cfg, 'num_key_value_heads', 32),
            "head_dim": getattr(cfg, 'hidden_size', 4096) // getattr(cfg, 'num_attention_heads', 32),
            "vocab_size": getattr(cfg, 'vocab_size', 151936),
            "layer_num": getattr(cfg, 'num_hidden_layers', 28),
            "intermediate_size": getattr(cfg, 'intermediate_size', 12288),
            "hidden_size": getattr(cfg, 'hidden_size', 4096),
            "norm_eps": getattr(cfg, 'rms_norm_eps', 1e-6),
            "max_batch_size": engine_config.max_batch_size,
            "max_context_len": engine_config.session_len,
            "cache_block_seq_len": engine_config.cache_block_seq_len,
            "rope_theta": getattr(cfg, 'rope_theta', 1000000),
            "rope_scaling": getattr(cfg, 'rope_scaling', None),
            "quant_policy": engine_config.quant_policy,
            "start": True,
        }
    else:
        # Fallback
        model_config = {
            "model_name": "exported_model",
            "tensor_para_size": engine_config.tp,
            "head_num": 32,
            "kv_head_num": 32,
            "head_dim": 128,
            "vocab_size": 151936,
            "layer_num": 28,
            "intermediate_size": 12288,
            "hidden_size": 4096,
            "norm_eps": 1e-6,
            "max_batch_size": engine_config.max_batch_size,
            "max_context_len": engine_config.session_len,
            "cache_block_seq_len": engine_config.cache_block_seq_len,
            "quant_policy": engine_config.quant_policy,
            "start": True,
        }

    with open(config_path, "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    print(f"[AWQ Converter] Wrote {config_path}")


def main():
    args = parse_args()
    export_weights_via_turbomind(
        args.model_path,
        args.output_dir,
        args.session_len,
        args.tp,
        args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
