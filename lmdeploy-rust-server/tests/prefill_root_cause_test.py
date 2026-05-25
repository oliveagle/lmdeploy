#!/usr/bin/env python3
"""
根因分析测试：对比 Python 和 Rust 的 C++ TurboMind 调用

测试目标：
1. 确认 Python 的 42K tok/s 性能是否真实
2. 找出 Rust 比 Python 慢的原因
3. 定位具体的性能瓶颈

模型: Qwen3.6-35B-A3B-AWQ
"""

import os
import sys
import time
import json
from pathlib import Path

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

sys.path.insert(0, str(Path(__file__).parent.parent / "lmdeploy" / "lib"))

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
REPEAT_TEXT = "The quick brown fox jumps over the lazy dog. "

def gen_prompt(token_count: int) -> str:
    repeats = max(1, token_count // 10)
    return REPEAT_TEXT * repeats


def test_tokenizer_performance(tokenizer, prompt, runs=5):
    """测试 tokenizer 编码性能"""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        ids = tokenizer.encode(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {
        'avg_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'token_count': len(tokenizer.encode(prompt))
    }


def test_direct_cpp_performance_sync(tm, input_ids, session_start=1000, runs=5):
    """直接测试 C++ TurboMind 的 prefill 性能"""
    import asyncio

    async def run_one(r):
        inst = tm.create_instance()
        session_id = r + session_start
        start = time.perf_counter()
        try:
            async for out in inst.async_stream_infer(
                session_id=session_id,
                input_ids=input_ids,
                gen_config=GenerationConfig(max_new_tokens=1),
                sequence_start=True,
                sequence_end=True,
            ):
                if out.status.value in (1, 2):
                    return (time.perf_counter() - start) * 1000
        except Exception as e:
            print(f"Error: {e}")
            return None

    return asyncio.run(run_one(0))


def test_direct_cpp_performance(tm, input_ids, runs=5):
    """直接测试 C++ TurboMind 的 prefill 性能"""
    times = []
    for r in range(runs):
        t = test_direct_cpp_performance_sync(tm, input_ids, session_start=1000+r, runs=1)
        if t:
            times.append(t)
    return times


def test_with_pipeline(pipe, prompt, runs=5):
    """使用 pipeline.stream_infer 测试"""
    times = []
    
    for r in range(runs):
        start = time.perf_counter()
        ttft = None
        
        for response in pipe.stream_infer(
            [prompt],
            gen_config=GenerationConfig(max_new_tokens=1, temperature=0.7),
            do_preprocess=False,
            stream_response=True,
        ):
            if ttft is None and response.generate_token_len:
                ttft = time.perf_counter() - start
        
        if ttft:
            times.append(ttft * 1000)
    
    return times


async def main():
    print("=" * 80)
    print("根因分析测试：Python vs Rust TurboMind C++ 调用")
    print("=" * 80)
    
    # 加载模型
    print("\n[1/5] 加载 TurboMind 引擎...")
    tm = TurboMind(
        model_path=MODEL,
        engine_config=TurbomindEngineConfig(
            session_len=32768,
            max_batch_size=1,
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
            max_prefill_token_num=32768,
            model_format='awq',
            cache_max_entry_count=0.4,
        ),
    )
    print("引擎加载成功")
    
    # 加载 tokenizer
    print("\n[2/5] 加载 Tokenizer...")
    tok = Tokenizer(MODEL)
    print(f"Tokenizer 加载成功, vocab_size: {tok.count_tokens()}")
    
    # 创建 pipeline（用于对比）
    print("\n[3/5] 创建 Pipeline（对比用）...")
    from lmdeploy import pipeline
    pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(
        session_len=32768, max_batch_size=1, tp=1, model_format='awq'
    ), log_level='ERROR')
    print("Pipeline 创建成功")
    
    # 生成测试 prompt
    test_cases = [
        ("1K", 1024),
        ("4K", 4096),
        ("8K", 8192),
    ]
    
    results = {}
    
    for label, target_tokens in test_cases:
        print(f"\n{'=' * 80}")
        print(f"测试 {label} context ({target_tokens} tokens)")
        print("=" * 80)
        
        prompt = gen_prompt(target_tokens)
        input_ids = tok.encode(prompt)
        actual_tokens = len(input_ids)
        print(f"实际 token 数: {actual_tokens}")
        
        # Test 1: Tokenizer 性能
        print(f"\n[Test 1] Tokenizer 编码性能...")
        tok_result = test_tokenizer_performance(tok, prompt, runs=5)
        print(f"  平均: {tok_result['avg_ms']:.2f}ms, 最小: {tok_result['min_ms']:.2f}ms")
        
        # Warmup
        print(f"\n[Warmup] 2 runs...")
        warmup_ids = tok.encode(gen_prompt(1024))
        for i in range(2):
            t = test_direct_cpp_performance_sync(tm, warmup_ids, session_start=i)
        print("Warmup 完成")
        
        # Test 2: 直接 C++ TurboMind 调用
        print(f"\n[Test 2] 直接 C++ TurboMind 调用...")
        cpp_times = test_direct_cpp_performance(tm, input_ids, runs=3)
        if cpp_times:
            avg_tps = actual_tokens / (sum(cpp_times) / len(cpp_times)) * 1000
            print(f"  TTFTs: {[f'{t:.1f}ms' for t in cpp_times]}")
            print(f"  平均 TPS: {avg_tps:.0f}")
        else:
            print("  C++ 测试失败")
            cpp_times = []
        
        # Test 3: Pipeline stream_infer
        print(f"\n[Test 3] Pipeline.stream_infer...")
        pipe_times = test_with_pipeline(pipe, prompt, runs=3)
        if pipe_times:
            avg_tps = actual_tokens / (sum(pipe_times) / len(pipe_times)) * 1000
            print(f"  TTFTs: {[f'{t:.1f}ms' for t in pipe_times]}")
            print(f"  平均 TPS: {avg_tps:.0f}")
        else:
            print("  Pipeline 测试失败")
            pipe_times = []
        
        results[label] = {
            'tokens': actual_tokens,
            'tok_avg_ms': tok_result['avg_ms'],
            'cpp_times_ms': cpp_times,
            'cpp_avg_ms': sum(cpp_times) / len(cpp_times) if cpp_times else 0,
            'pipe_times_ms': pipe_times,
            'pipe_avg_ms': sum(pipe_times) / len(pipe_times) if pipe_times else 0,
        }
    
    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"\n{'Context':>8} | {'Tokens':>8} | {'Tokenizer':>12} | {'C++ Direct':>12} | {'Pipeline':>12}")
    print("-" * 60)
    
    for label in ["1K", "4K", "8K"]:
        r = results[label]
        cpp_tps = r['tokens'] / r['cpp_avg_ms'] * 1000 if r['cpp_avg_ms'] > 0 else 0
        pipe_tps = r['tokens'] / r['pipe_avg_ms'] * 1000 if r['pipe_avg_ms'] > 0 else 0
        print(f"{label:>8} | {r['tokens']:>8} | {r['tok_avg_ms']:>10.2f}ms | {cpp_tps:>10.0f} tok/s | {pipe_tps:>10.0f} tok/s")
    
    # 保存结果
    output_path = Path(__file__).parent / "prefill_root_cause_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {output_path}")
    
    tm.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
