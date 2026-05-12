#!/usr/bin/env python3
"""
DFlash 推理测试 - 验证基准测试仍然工作
"""

import os
import time

os.environ['LD_LIBRARY_PATH'] = f'/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"

DURATION = 5  # 运行 5 秒

gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)

prompts = [
    "Python 是什么？",
]

def main():
    print("=" * 60)
    print("DFlash 推理测试 - 基准测试 (禁用 speculative)")
    print("=" * 60)
    print(f"目标: {target_model}")
    print(f"运行时长: {DURATION} 秒\n")

    # 禁用 speculative decoding
    speculative_config = None

    tm_config = TurbomindEngineConfig(
        model_format='awq',
        tensor_parallel=1,
        cache_max_entry_count=0.2,
        quant_policy=8,
        session_len=16384,
    )

    print("创建 Pipeline...")
    pipe = pipeline(
        target_model,
        backend_config=tm_config,
        speculative_config=speculative_config,
        log_level='WARNING'
    )
    print("✓ 创建成功！\n")

    start_time = time.time()
    total_tokens = 0
    total_requests = 0
    prompt_idx = 0

    print("开始推理...")

    while time.time() - start_time < DURATION:
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1

        t0 = time.time()
        resp = pipe(
            [{"role": "user", "content": prompt}],
            gen_config=gen_config,
            sequence_start=True,
            sequence_end=True,
            chat_template_kwargs={'enable_thinking': False}
        )
        t1 = time.time()

        elapsed = t1 - t0
        output_tokens = len(resp.token_ids) - resp.input_token_len
        tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0
        total_tokens += output_tokens
        total_requests += 1

        print(f"Request {total_requests}: {output_tokens} tokens in {elapsed:.2f}s, {tokens_per_sec:.2f} tok/s")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("基准测试结果")
    print("=" * 60)
    print(f"总请求数: {total_requests}")
    print(f"总 Tokens: {total_tokens}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均耗时: {total_time/total_requests:.3f}s/请求")
    print(f"平均速度: {total_tokens/total_time:.2f} tokens/s")
    print("=" * 60)

if __name__ == "__main__":
    main()
