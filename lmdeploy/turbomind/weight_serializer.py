#!/usr/bin/env python3
"""
TurboMind Weight Serializer

Converts HuggingFace safetensors to TurboMind .bin format for C API loading.

This script:
1. Reads HF safetensors weights
2. Converts them to TurboMind format
3. Serializes to .bin files that InitFromPath() can load

Usage:
    python weight_serializer.py <model_path> <output_dir>

Reference:
    - Model weight format: src/turbomind/models/model_weight.h
    - C API InitFromPath: src/turbomind/capi/turbomind_c.cc
"""

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open


@dataclass
class TensorInfo:
    """Metadata for a serialized tensor."""
    name: str
    dtype: str  # 'fp16', 'fp32', 'int4', etc.
    shape: Tuple[int, ...]
    offset: int  # Byte offset in the .bin file
    size: int  # Size in bytes


@dataclass
class WeightFile:
    """A .bin file containing multiple tensors."""
    path: str
    tensors: List[TensorInfo] = field(default_factory=list)
    current_offset: int = 0

    def add_tensor(self, name: str, tensor: torch.Tensor) -> TensorInfo:
        """Add a tensor to this file and return its info."""
        dtype = self._torch_dtype_to_str(tensor.dtype)
        shape = tuple(tensor.shape)

        # Calculate size
        size = tensor.numel() * tensor.element_size()

        # Align to 64-byte boundary
        aligned_offset = (self.current_offset + 63) & ~63
        if aligned_offset > self.current_offset:
            padding = aligned_offset - self.current_offset
            self.current_offset = aligned_offset

        info = TensorInfo(
            name=name,
            dtype=dtype,
            shape=shape,
            offset=self.current_offset,
            size=size
        )
        self.tensors.append(info)
        self.current_offset += size
        return info

    @staticmethod
    def _torch_dtype_to_str(dtype: torch.dtype) -> str:
        mapping = {
            torch.float16: 'fp16',
            torch.float32: 'fp32',
            torch.bfloat16: 'bf16',
            torch.int32: 'int32',
            torch.int8: 'int8',
            torch.uint8: 'uint8',
        }
        return mapping.get(dtype, 'unknown')


class WeightSerializer:
    """Serializes HF weights to TurboMind .bin format."""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(self.model_path / "config.json") as f:
            self.config = json.load(f)

        # Handle nested config (e.g., Qwen3.5 MoE has text_config)
        config_source = self.config
        if "text_config" in self.config:
            config_source = self.config["text_config"]
        elif "model_config" in self.config:
            config_source = self.config["model_config"]

        # Model parameters
        self.hidden_size = config_source.get("hidden_size", self.config.get("hidden_size", 4096))
        self.num_layers = config_source.get("num_hidden_layers", self.config.get("num_hidden_layers", 32))
        self.num_heads = config_source.get("num_attention_heads", self.config.get("num_attention_heads", 32))
        self.num_kv_heads = config_source.get("num_key_value_heads", self.config.get("num_key_value_heads", self.num_heads))
        self.vocab_size = config_source.get("vocab_size", self.config.get("vocab_size", 32000))
        self.intermediate_size = config_source.get("intermediate_size", self.config.get("intermediate_size", self.hidden_size * 4))

        # Check for AWQ quantization
        quant_config = self.config.get("quantization_config", {})
        self.is_awq = quant_config.get("quant_method") == "awq"
        self.group_size = quant_config.get("group_size", 128) if self.is_awq else None

        # Find safetensors files
        self.safetensors_files = self._find_safetensors()

        # Weight files and tensor index
        self.weight_files: List[WeightFile] = []
        self.tensor_index: Dict[str, Tuple[int, int]] = {}  # name -> (file_idx, tensor_idx)

    def _find_safetensors(self) -> List[Path]:
        """Find all safetensors files."""
        index_file = self.model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            files = sorted(set(index["weight_map"].values()))
            return [self.model_path / f for f in files]

        # Fallback: glob for .safetensors files
        return sorted(self.model_path.glob("*.safetensors"))

    def serialize(self) -> Dict[str, Any]:
        """Serialize all weights to .bin files."""
        print(f"[Serializer] Processing {len(self.safetensors_files)} safetensors files...")

        # Create weight files (one per safetensors shard for now)
        for i, st_file in enumerate(self.safetensors_files):
            bin_path = self.output_dir / f"weights_{i}.bin"
            weight_file = WeightFile(path=str(bin_path))
            self.weight_files.append(weight_file)

            # Process tensors from this safetensors file
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    info = weight_file.add_tensor(name, tensor)
                    file_idx = len(self.weight_files) - 1
                    tensor_idx = len(weight_file.tensors) - 1
                    self.tensor_index[name] = (file_idx, tensor_idx)

        # Write the .bin files
        self._write_bin_files()

        # Write config.yaml
        self._write_config_yaml()

        # Write weight index
        self._write_weight_index()

        # Return summary
        return {
            "num_files": len(self.weight_files),
            "num_tensors": sum(len(wf.tensors) for wf in self.weight_files),
            "total_size": sum(wf.current_offset for wf in self.weight_files),
        }

    def _write_bin_files(self):
        """Write all tensors to .bin files."""
        print(f"[Serializer] Writing {len(self.weight_files)} .bin files...")

        # Map tensor names to their source safetensors
        tensor_map: Dict[str, Tuple[Path, str]] = {}
        for st_file in self.safetensors_files:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor_map[name] = (st_file, name)

        # Write each .bin file
        for weight_file in self.weight_files:
            with open(weight_file.path, "wb") as f:
                for tensor_info in weight_file.tensors:
                    # Get tensor from source
                    st_path, st_name = tensor_map[tensor_info.name]
                    with safe_open(st_path, framework="pt", device="cpu") as st:
                        tensor = st.get_tensor(st_name)

                    # Convert to contiguous and write
                    tensor = tensor.contiguous()
                    # Pad to aligned offset if needed
                    current_pos = f.tell()
                    if current_pos < tensor_info.offset:
                        f.write(b"\x00" * (tensor_info.offset - current_pos))

                    # Write tensor data
                    # Handle different dtypes
                    if tensor.dtype == torch.bfloat16:
                        # BF16 needs special handling - convert to float32 then to uint16
                        data = tensor.float().numpy().view(np.uint16).tobytes()
                    elif tensor.dtype == torch.float16:
                        data = tensor.numpy().view(np.uint16).tobytes()
                    elif tensor.dtype == torch.float32:
                        data = tensor.numpy().view(np.uint32).tobytes()
                    else:
                        data = tensor.numpy().tobytes()
                    f.write(data)

            print(f"[Serializer] Wrote {weight_file.path} ({weight_file.current_offset} bytes)")

    def _write_config_yaml(self):
        """Write TurboMind config.yaml."""
        config_path = self.output_dir / "config.yaml"

        # Basic config structure
        config = {
            "model_name": "qwen",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "intermediate_size": self.intermediate_size,
        }

        # AWQ quantization config
        if self.is_awq:
            config["quantization"] = {
                "type": "awq",
                "group_size": self.group_size,
                "bits": 4,
            }

        # Write as YAML
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"[Serializer] Wrote {config_path}")

    def _write_weight_index(self):
        """Write weight index JSON."""
        index_path = self.output_dir / "weight_index.json"

        index = {
            "metadata": {
                "total_files": len(self.weight_files),
                "total_tensors": sum(len(wf.tensors) for wf in self.weight_files),
            },
            "weight_map": {},
            "files": [wf.path for wf in self.weight_files],
        }

        # Build weight map
        for weight_file in self.weight_files:
            for tensor_info in weight_file.tensors:
                index["weight_map"][tensor_info.name] = {
                    "file": weight_file.path,
                    "offset": tensor_info.offset,
                    "dtype": tensor_info.dtype,
                    "shape": list(tensor_info.shape),
                    "size": tensor_info.size,
                }

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[Serializer] Wrote {index_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Serialize HF weights to TurboMind .bin format")
    parser.add_argument("model_path", help="Path to HuggingFace model (safetensors)")
    parser.add_argument("output_dir", help="Output directory for .bin files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--num-shards", "-n", type=int, default=None,
                        help="Number of output shards (default: same as input)")

    args = parser.parse_args()

    serializer = WeightSerializer(args.model_path, args.output_dir)
    summary = serializer.serialize()

    print(f"[Serializer] Done! Summary:")
    print(f"  Files: {summary['num_files']}")
    print(f"  Tensors: {summary['num_tensors']}")
    print(f"  Total size: {summary['total_size'] / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
