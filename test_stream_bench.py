#!/usr/bin/env python3
"""Quick streaming benchmark for LMDeploy."""

import time
import requests
import json

def test_streaming(url="http://localhost:8000", model="Qwen3.6-35B-A3B-AWQ", max_tokens=128):
    """Test streaming performance."""
    headers = {"Content-Type": "application/json"}

    # Generate a 512-token prompt (approx 2048 chars)
    prompt = "Hello " * 400  # ~2000 chars, ~500 tokens

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }

    start = time.time()
    first_token_time = None
    token_count = 0
    last_chunk_time = None

    response = requests.post(f"{url}/v1/chat/completions", headers=headers, json=data, stream=True, timeout=300)

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            if first_token_time is None:
                                first_token_time = time.time()
                            token_count += 1
                            last_chunk_time = time.time()
                except json.JSONDecodeError:
                    pass

    total_time = time.time() - start

    if first_token_time:
        ttft = (first_token_time - start) * 1000
        decode_time = last_chunk_time - first_token_time if last_chunk_time else 0
        tps = token_count / decode_time if decode_time > 0 else 0

        print(f"Total time: {total_time:.2f}s")
        print(f"Time to first token: {ttft:.2f}ms")
        print(f"Tokens received: {token_count}")
        print(f"Decode time: {decode_time:.2f}s")
        print(f"Decode speed: {tps:.2f} tokens/s")
    else:
        print("No tokens received!")

    return token_count > 0

if __name__ == "__main__":
    test_streaming()
