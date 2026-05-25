#!/usr/bin/env python3
"""
全面分析 pipeline.stream_infer 的 TTFT 测量机制

测试目标：
1. 理解 pipeline.stream_infer 的响应模式
2. 找到正确的 TTFT 测量方法
3. 对比 stream_output=True vs stream_output=False
4. 确认历史 42K tok/s 是如何测量的
"""

import os
import sys
import time
import asyncio
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
PROMPT_8K = "The quick brown fox jumps over the lazy dog. " * 820


def test_pipeline_stream_response_true():
    """测试 stream_response=True 模式"""
    print("=" * 80)
    print("测试 1: pipeline.stream_infer(stream_response=True)")
    print("=" * 80)

    pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(
        session_len=16384, max_batch_size=1, tp=1, model_format='awq', cache_max_entry_count=0.4,
    ), log_level='ERROR')

    # Warmup
    for _ in range(1):
        for _ in pipe.stream_infer(["hello"], gen_config=GenerationConfig(max_new_tokens=5), do_preprocess=False, stream_response=True):
            pass

    print("\n8K Prompt 测试...")
    start = time.perf_counter()
    responses = []
    for response in pipe.stream_infer(
        [PROMPT_8K],
        gen_config=GenerationConfig(max_new_tokens=10, temperature=0.7, do_sample=False),
        do_preprocess=False,
        stream_response=True,
    ):
        elapsed = (time.perf_counter() - start) * 1000
        responses.append({
            'elapsed_ms': elapsed,
            'gen_token_len': response.generate_token_len,
            'text_len': len(response.text),
            'finish_reason': response.finish_reason,
        })
        print(f"  t={elapsed:>7.1f}ms: gen_token_len={response.generate_token_len:>3}, text_len={len(response.text):>4}, finish={response.finish_reason}")
        if response.finish_reason:
            break

    print(f"\n总共收到 {len(responses)} 个响应")
    if responses:
        first_with_tokens = [r for r in responses if r['gen_token_len'] > 0]
        if first_with_tokens:
            print(f"第一个有 token 的响应: t={first_with_tokens[0]['elapsed_ms']:.1f}ms")
        else:
            print("没有收到任何有 token 的响应！")

    pipe.close()
    return responses


async def test_turbomind_stream_output_false():
    """测试 TurboMind 直接调用，stream_output=False"""
    print("\n" + "=" * 80)
    print("测试 2: TurboMind.async_stream_infer(stream_output=False)")
    print("=" * 80)

    tm = TurboMind(MODEL, engine_config=TurbomindEngineConfig(
        session_len=16384, max_batch_size=1, tp=1, model_format='awq', cache_max_entry_count=0.4,
    ))
    tok = Tokenizer(MODEL)
    inst = tm.create_instance()

    # Warmup
    warmup_ids = tok.encode("hello")
    for _ in range(1):
        async for _ in inst.async_stream_infer(
            session_id=0,
            input_ids=warmup_ids,
            gen_config=GenerationConfig(max_new_tokens=5),
            sequence_start=True,
            sequence_end=True,
        ):
            pass

    print("\n8K Prompt 测试...")
    input_ids = tok.encode(PROMPT_8K)
    print(f"Token count: {len(input_ids)}")

    start = time.perf_counter()
    outputs = []
    async for output in inst.async_stream_infer(
        session_id=1,
        input_ids=input_ids,
        gen_config=GenerationConfig(max_new_tokens=10, temperature=0.7, do_sample=False),
        sequence_start=True,
        sequence_end=True,
        stream_output=False,  # 关键：等待所有 token 生成完
    ):
        elapsed = (time.perf_counter() - start) * 1000
        outputs.append({
            'elapsed_ms': elapsed,
            'token_count': len(output.token_ids),
            'status': output.status,
        })
        print(f"  t={elapsed:>7.1f}ms: tokens={len(output.token_ids):>3}, status={output.status}")

    print(f"\n总共收到 {len(outputs)} 个输出")
    if outputs:
        print(f"TTFT (第一次输出): {outputs[0]['elapsed_ms']:.1f}ms")
        print(f"Token count: {outputs[0]['token_count']}")

    tm.close()
    return outputs


async def test_turbomind_stream_output_true():
    """测试 TurboMind 直接调用，stream_output=True"""
    print("\n" + "=" * 80)
    print("测试 3: TurboMind.async_stream_infer(stream_output=True)")
    print("=" * 80)

    tm = TurboMind(MODEL, engine_config=TurbomindEngineConfig(
        session_len=16384, max_batch_size=1, tp=1, model_format='awq', cache_max_entry_count=0.4,
    ))
    tok = Tokenizer(MODEL)
    inst = tm.create_instance()

    # Warmup
    warmup_ids = tok.encode("hello")
    for _ in range(1):
        async for _ in inst.async_stream_infer(
            session_id=10,
            input_ids=warmup_ids,
            gen_config=GenerationConfig(max_new_tokens=5),
            sequence_start=True,
            sequence_end=True,
        ):
            pass

    print("\n8K Prompt 测试...")
    input_ids = tok.encode(PROMPT_8K)

    start = time.perf_counter()
    outputs = []
    async for output in inst.async_stream_infer(
        session_id=11,
        input_ids=input_ids,
        gen_config=GenerationConfig(max_new_tokens=10, temperature=0.7, do_sample=False),
        sequence_start=True,
        sequence_end=True,
        stream_output=True,  # 关键：token-by-token 输出
    ):
        elapsed = (time.perf_counter() - start) * 1000
        outputs.append({
            'elapsed_ms': elapsed,
            'token_count': len(output.token_ids),
            'status': output.status,
        })
        print(f"  t={elapsed:>7.1f}ms: tokens={len(output.token_ids):>3}, status={output.status}")
        if output.status.value == 7:  # FINISH
            break

    print(f"\n总共收到 {len(outputs)} 个输出")
    if outputs:
        print(f"TTFT (第一个 token): {outputs[0]['elapsed_ms']:.1f}ms")
        print(f"Token count in first output: {outputs[0]['token_count']}")

    tm.close()
    return outputs


async def main():
    print("\n" + "=" * 80)
    print("LMDeploy TTFT 测量机制分析")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Prompt: 8K tokens (approx)")
    print()

    # Test 1: Pipeline with stream_response=True
    resp1 = test_pipeline_stream_response_true()

    # Test 2: TurboMind with stream_output=False
    resp2 = await test_turbomind_stream_output_false()

    # Test 3: TurboMind with stream_output=True
    resp3 = await test_turbomind_stream_output_true()

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"\n{'方法':<40} | {'TTFT (ms)':<12} | {'备注':<20}")
    print("-" * 80)

    if resp1:
        first_with_tokens = [r for r in resp1 if r['gen_token_len'] > 0]
        if first_with_tokens:
            ttft = first_with_tokens[0]['elapsed_ms']
            print(f"{'Pipeline(stream_response=True)':<40} | {ttft:>10.1f}  | {'首次有 token':<20}")
        else:
            print(f"{'Pipeline(stream_response=True)':<40} | {'N/A':>10}  | {'无 token 输出':<20}")

    if resp2:
        ttft = resp2[0]['elapsed_ms']
        print(f"{'TurboMind(stream_output=False)':<40} | {ttft:>10.1f}  | {'等待全部完成':<20}")

    if resp3:
        ttft = resp3[0]['elapsed_ms']
        print(f"{'TurboMind(stream_output=True)':<40} | {ttft:>10.1f}  | {'token-by-token':<20}")


if __name__ == "__main__":
    asyncio.run(main())
