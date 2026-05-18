#!/usr/bin/env python3
"""
TurboMind Python Bridge - Provides inference access to TurboMind Python API.

This module wraps the Python TurboMind API and exposes it via stdin/stdout
for use by the Rust server. This bypasses the C API's broken weight loading.

Usage:
    python3 -m lmdeploy.turbomind.python_bridge <model_path> [--session-len N] [--tp N]

Protocol:
    - Reads JSON commands from stdin (one per line)
    - Writes JSON responses to stdout (one per line)
    - Commands:
        - {"cmd": "ping"} -> {"status": "ok", "pong": true}
        - {"cmd": "load", "model_path": "...", "engine_config": {...}} -> {"status": "ok"} or {"status": "error", "message": "..."}
        - {"cmd": "generate", "input_ids": [1,2,3], "max_new_tokens": 100} -> {"status": "ok", "output_ids": [...]}
        - {"cmd": "metrics"} -> {"status": "ok", "metrics": {...}}
        - {"cmd": "shutdown"} -> {"status": "ok"}
"""

import argparse
import asyncio
import json
import os.path as osp
import sys


def setup_paths():
    """Setup Python paths for lmdeploy imports."""
    import lmdeploy
    lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
    sys.path.insert(0, osp.join(lmdeploy_dir, 'lib'))


setup_paths()

import logging
import tempfile

# Redirect lmdeploy logs to stderr instead of stdout
_logger = logging.getLogger('lmdeploy')
for handler in _logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
        handler.stream = sys.stderr

import numpy as np
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind


class TurboMindBridge:
    """Python bridge for TurboMind inference."""

    def __init__(self):
        self.tm: TurboMind | None = None
        self.instance = None
        self.model_path: str | None = None
        self.device_id: int = 0
        self.is_loaded: bool = False
        self._session_id: int = 1

    def load_model(self, model_path: str, engine_config: dict | None = None) -> dict:
        """Load a model via TurboMind Python API."""
        if self.is_loaded:
            return {"status": "error", "message": "Model already loaded. Use shutdown first."}

        try:
            print(f"[Bridge] Loading model from {model_path}", file=sys.stderr)

            # Build engine config - set tp explicitly, let Pydantic defaults handle the rest
            tp = engine_config.get("tp", 1) if engine_config else 1
            ec = TurbomindEngineConfig(
                session_len=engine_config.get("session_len", 2048) if engine_config else 2048,
                max_batch_size=engine_config.get("max_batch_size", 32) if engine_config else 32,
                cache_block_seq_len=engine_config.get("cache_block_seq_len", 64) if engine_config else 64,
                cache_max_entry_count=engine_config.get("cache_max_entry_count", 0.8) if engine_config else 0.8,
                tp=tp,
                dp=1,
                cp=1,
                enable_prefix_caching=False,  # Disable prefix caching for linear attention models
                enable_metrics=True,
                # quant_policy is set after creation due to Pydantic constraints
            )
            if engine_config and engine_config.get("quant_policy"):
                ec.quant_policy = engine_config["quant_policy"]

            # Create TurboMind instance - this loads weights via Python API
            # The _from_hf -> model_loader.export() -> _process_weights() -> _create_engine()
            # pipeline is fully implemented in Python
            self.tm = TurboMind(
                model_path=model_path,
                engine_config=ec,
                trust_remote_code=True,
            )

            self.model_path = model_path
            self.is_loaded = True

            # Create an instance for inference
            self.instance = self.tm.create_instance(cuda_stream_id=0)

            print(f"[Bridge] Model loaded successfully", file=sys.stderr)
            return {
                "status": "ok",
                "model_path": model_path,
                "session_len": ec.session_len,
            }

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}

    async def generate(self, input_ids: list[int], max_new_tokens: int = 100,
                       temperature: float = 0.7, top_p: float = 0.95,
                       top_k: int = 50) -> dict:
        """Generate tokens from input_ids."""
        if not self.is_loaded or not self.instance:
            return {"status": "error", "message": "Model not loaded"}

        try:
            import time

            # Create generation config
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # Convert input_ids to numpy array
            input_ids_array = np.array(input_ids, dtype=np.int32)

            # Run inference via TurboMind instance
            start = time.time()
            output_ids = []

            # Use async_stream_infer and collect all output
            async for output in self.instance.async_stream_infer(
                session_id=self._session_id,
                input_ids=input_ids,
                gen_config=gen_config,
                sequence_start=True,
                sequence_end=True,
                stream_output=False,
            ):
                if output.status == 0:  # SUCCESS
                    output_ids.extend(output.token_ids)

            self._session_id += 1
            elapsed = time.time() - start

            return {
                "status": "ok",
                "output_ids": output_ids,
                "elapsed_ms": round(elapsed * 1000, 2),
            }

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}

    async def batch_generate(self, requests: list[dict]) -> dict:
        """Batch generate for multiple requests."""
        if not self.is_loaded:
            return {"status": "error", "message": "Model not loaded"}

        try:
            results = []
            for req in requests:
                result = await self.generate(
                    input_ids=req.get("input_ids", []),
                    max_new_tokens=req.get("max_new_tokens", 100),
                    temperature=req.get("temperature", 0.7),
                    top_p=req.get("top_p", 0.95),
                    top_k=req.get("top_k", 50),
                )
                results.append(result)

            return {"status": "ok", "results": results}

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}

    def get_metrics(self) -> dict:
        """Get schedule metrics from TurboMind."""
        if not self.is_loaded or not self.tm:
            return {"status": "error", "message": "Model not loaded"}

        try:
            metrics = self.tm.get_schedule_metrics()
            if metrics:
                return {
                    "status": "ok",
                    "metrics": {
                        "active_seqs": metrics.active_seqs,
                        "waiting_seqs": metrics.waiting_seqs,
                        "total_blocks": metrics.total_blocks,
                        "active_blocks": metrics.active_blocks,
                        "free_blocks": metrics.free_blocks,
                    }
                }
            return {"status": "ok", "metrics": {}}

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}

    def shutdown(self) -> dict:
        """Shutdown the bridge."""
        print("[Bridge] Shutting down...", file=sys.stderr)
        if self.tm:
            self.tm.close()
            self.tm = None
        self.is_loaded = False
        return {"status": "ok"}


async def main_async():
    """Async main entry point for the bridge."""
    parser = argparse.ArgumentParser(description="TurboMind Python Bridge")
    parser.add_argument("--model-path", type=str, help="Model path to load at startup")
    parser.add_argument("--session-len", type=int, default=2048, help="Max context length")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--quant-policy", type=int, default=0, help="Quantization policy (4 for AWQ)")
    args = parser.parse_args()

    bridge = TurboMindBridge()

    # Preload model if path provided
    if args.model_path:
        result = bridge.load_model(args.model_path, {
            "session_len": args.session_len,
            "tp": args.tp,
            "quant_policy": args.quant_policy,
            "max_batch_size": 32,
            "cache_block_seq_len": 64,
        })
        print(json.dumps(result), flush=True)
        if result.get("status") != "ok":
            print(json.dumps({"status": "error", "message": f"Failed to load model: {result.get('message')}"}), flush=True)
            sys.exit(1)

    # Main command loop
    print("[Bridge] Ready for commands", file=sys.stderr)
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Read from stdin asynchronously
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break  # EOF

            line = line.strip()
            if not line:
                continue

            cmd = json.loads(line)
            cmd_name = cmd.get("cmd")

            if cmd_name == "ping":
                response = {"status": "ok", "pong": True}
            elif cmd_name == "load":
                model_path = cmd.get("model_path")
                engine_config = cmd.get("engine_config", {})
                response = bridge.load_model(model_path, engine_config)
            elif cmd_name == "generate":
                response = await bridge.generate(
                    input_ids=cmd.get("input_ids", []),
                    max_new_tokens=cmd.get("max_new_tokens", 100),
                    temperature=cmd.get("temperature", 0.7),
                    top_p=cmd.get("top_p", 0.95),
                    top_k=cmd.get("top_k", 50),
                )
            elif cmd_name == "batch_generate":
                response = await bridge.batch_generate(cmd.get("requests", []))
            elif cmd_name == "metrics":
                response = bridge.get_metrics()
            elif cmd_name == "shutdown":
                response = bridge.shutdown()
                print(json.dumps(response), flush=True)
                break
            else:
                response = {"status": "error", "message": f"Unknown command: {cmd_name}"}

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"status": "error", "message": str(e)}), flush=True)


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
