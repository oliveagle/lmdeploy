#!/usr/bin/env python3
"""Direct test of Python TurboMind inference with Qwen3.6-35B-A3B-AWQ."""

import os
import json

MODEL_PATH = "/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ"

def test_inference():
    from lmdeploy.turbomind.turbomind import TurboMind
    from lmdeploy.messages import TurbomindEngineConfig, GenerationConfig
    from lmdeploy.tokenizer import Tokenizer
    import asyncio

    # Load model
    print("Loading model...")
    engine_config = TurbomindEngineConfig(
        session_len=2048,
        max_batch_size=8,
        quant_policy=4,
        empty_init=False,
    )

    tm = TurboMind(
        model_path=MODEL_PATH,
        engine_config=engine_config,
        trust_remote_code=True,
    )
    print("Model loaded.")

    tokenizer = Tokenizer(MODEL_PATH, trust_remote_code=True)
    tm_instance = tm.create_instance()

    # Generate
    async def gen():
        prompt = "Hello, what is 2+2?"
        input_ids = tokenizer.encode(prompt)
        print(f"Input tokens: {len(input_ids)}")

        gen_config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
        )

        output_tokens = []
        session_id = 1

        async for out in tm_instance.async_stream_infer(
            session_id=session_id,
            input_ids=input_ids,
            gen_config=gen_config,
            stream_output=True,
            sequence_start=True,
            sequence_end=True,
            step=0,
        ):
            print(f"  Step: status={out.status}, status_name={out.status.name}, tokens={len(out.token_ids)}")
            if out.status == 7:  # finish
                print("  Generation finished")
                break
            if out.token_ids:
                output_tokens.extend(out.token_ids)

        text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"Output text: {text}")
        return text

    try:
        result = asyncio.run(gen())
        print(f"\nFinal: '{result}'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
