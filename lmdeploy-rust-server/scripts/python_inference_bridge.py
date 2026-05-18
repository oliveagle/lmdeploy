#!/usr/bin/env python3
"""
LMDeploy Python Inference Bridge

This script loads a model using the Python LMDeploy Pipeline API and provides a
simple stdin/stdout JSON API for inference. The Rust server spawns this
as a subprocess and communicates via JSON messages.

Protocol:
- Input (stdin): {"action": "generate", "prompt": "...", "max_tokens": 123}
- Output (stdout): {"status": "ok", "output": "..."} or {"status": "error", "message": "..."}
"""

import argparse
import json
import sys
import os
import time


def load_model(model_path: str, session_len: int = 2048, max_batch_size: int = 8):
    """Load the model using Python LMDeploy Pipeline API."""
    from lmdeploy import Pipeline, TurbomindEngineConfig

    # Detect AWQ quantization
    is_awq = False
    config_json = os.path.join(model_path, "config.json")
    if os.path.exists(config_json):
        with open(config_json) as f:
            config = json.load(f)
            quant_config = config.get("quantization_config", {})
            is_awq = quant_config.get("quant_method") == "awq"

    # Configure engine
    quant_policy = 4 if is_awq else 0
    engine_config = TurbomindEngineConfig(
        session_len=session_len,
        max_batch_size=max_batch_size,
        quant_policy=quant_policy,
        cache_max_entry_count=0.8,
    )

    print(f"[Bridge] Loading model: {model_path}", file=sys.stderr)
    print(f"[Bridge] AWQ: {is_awq}, quant_policy: {quant_policy}", file=sys.stderr)

    # Create Pipeline instance
    pipeline = Pipeline(
        model_path,
        backend_config=engine_config,
        log_level="WARNING",
    )

    print(f"[Bridge] Model loaded successfully", file=sys.stderr)
    return pipeline


def generate(pipeline, prompt: str, max_tokens: int = 100) -> str:
    """Generate text using the Pipeline infer API."""
    from lmdeploy.messages import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
    )

    # Use non-stream infer API
    responses = pipeline.infer(
        prompt,
        gen_config=gen_config,
    )

    # Handle both list and single Response
    if isinstance(responses, list):
        resp = responses[0] if responses else None
    else:
        resp = responses

    if resp is not None and hasattr(resp, 'text'):
        return resp.text
    return ""


def main():
    parser = argparse.ArgumentParser(description="LMDeploy Python Inference Bridge")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("--session-len", type=int, default=2048, help="Session length")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size")
    args = parser.parse_args()

    try:
        # Load model
        pipeline = load_model(
            args.model_path,
            session_len=args.session_len,
            max_batch_size=args.max_batch_size,
        )

        # Process stdin/stdout commands
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                action = request.get("action")

                if action == "generate":
                    prompt = request.get("prompt", "")
                    max_tokens = request.get("max_tokens", 100)

                    output = generate(pipeline, prompt, max_tokens)

                    response = {
                        "status": "ok",
                        "output": output,
                    }
                    print(json.dumps(response))
                    sys.stdout.flush()

                elif action == "health":
                    response = {
                        "status": "ok",
                        "message": "ready",
                    }
                    print(json.dumps(response))
                    sys.stdout.flush()

                elif action == "exit":
                    print(f"[Bridge] Exiting...", file=sys.stderr)
                    break

                else:
                    response = {
                        "status": "error",
                        "message": f"Unknown action: {action}",
                    }
                    print(json.dumps(response))
                    sys.stdout.flush()

            except json.JSONDecodeError as e:
                response = {
                    "status": "error",
                    "message": f"Invalid JSON: {e}",
                }
                print(json.dumps(response))
                sys.stdout.flush()

            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                response = {
                    "status": "error",
                    "message": str(e),
                }
                print(json.dumps(response))
                sys.stdout.flush()

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"[Bridge] Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
