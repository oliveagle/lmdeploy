#!/usr/bin/env python3
"""
Python inference helper for Rust LMDeploy server.

This script handles model initialization and provides inference via stdin/stdout.
"""
import sys
import json
import os

# Add lmdeploy lib to path
lmdeploy_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.join(lmdeploy_dir, 'lmdeploy', 'lib'))

from lmdeploy.turbomind import TurboMind, TurbomindEngineConfig
from lmdeploy.tokenizer import Tokenizer


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: inference_helper.py <model_path>"}))
        sys.exit(1)

    model_path = sys.argv[1]

    # Configure engine
    engine_config = TurbomindEngineConfig(
        session_len=8192,
        max_batch_size=32,
        max_prefill_token_num=4096,
        cache_block_seq_len=16,
        tp=1,
    )

    # Initialize TurboMind
    tm = TurboMind.from_pretrained(
        model_path,
        engine_config=engine_config,
        trust_remote_code=True
    )

    # Create instance
    instance = tm.create_instance()

    # Initialize tokenizer
    tokenizer = Tokenizer(model_path, trust_remote_code=True)

    # Write ready status
    print(json.dumps({"status": "ready", "model_path": model_path}))
    sys.stdout.flush()

    # Process requests from stdin
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())

            if request.get("action") == "generate":
                input_ids = request.get("input_ids", [])
                max_tokens = request.get("max_tokens", 512)
                temperature = request.get("temperature", 1.0)
                top_p = request.get("top_p", 0.95)

                # Generate
                from lmdeploy.messages import GenerationConfig
                gen_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                outputs = []
                for output in instance.stream_infer(
                    session_id=request.get("session_id", 1),
                    input_ids=input_ids,
                    gen_config=gen_config,
                    sequence_start=True,
                    step=0,
                ):
                    outputs.append(output.token_ids)
                    if output.finish:
                        break

                # Decode output
                output_ids = []
                for chunk in outputs:
                    output_ids.extend(chunk)

                response_text = tokenizer.decode(output_ids)

                print(json.dumps({
                    "status": "success",
                    "output_ids": output_ids,
                    "text": response_text
                }))
                sys.stdout.flush()

            elif request.get("action") == "shutdown":
                break

        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
