#!/usr/bin/env python3
"""Python Direct vs Rust C++ Unified Performance Benchmark"""
import sys; sys.path.insert(0, '/mnt/data/lmdeploy')
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
import time, json

MODEL = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
OUTPUT_TOKENS = 64
RUNS = 3

def run_test(label, ec_kwargs):
    """Run benchmark with specific engine config."""
    print(f"\n{'='*60}\nTesting: {label}")
    print(f"Config: {ec_kwargs}")

    ec = TurbomindEngineConfig(
        session_len=ec_kwargs.get("session_len", 65536),
        max_batch_size=ec_kwargs.get("max_batch_size", 128),
        cache_block_seq_len=ec_kwargs.get("cache_block_seq_len", 64),
        tp=ec_kwargs.get("tp", 1),
        enable_prefix_caching=ec_kwargs.get("prefix_caching", False),
    )
    tm = TurboMind(model_path=MODEL, engine_config=ec, trust_remote_code=True)
    tok = Tokenizer(MODEL, trust_remote_code=True)
    inst = tm.create_instance()

    results = {}
    for ctx_label, prompt in [
        ("5tok", "Hello, how are you"),
        ("600tok", "The quick brown fox jumps over the lazy dog. " * 100),
        ("3000tok", "The quick brown fox jumps over the lazy dog. " * 500),
    ]:
        input_ids = tok.encode(prompt)
        gen_cfg = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.7)

        # Warmup
        for out in inst.async_stream_infer(
            session_id=0, input_ids=input_ids, gen_config=gen_cfg,
            sequence_start=True, sequence_end=True,
        ):
            pass

        # Measured runs
        run_data = []
        for r in range(RUNS):
            start = time.perf_counter()
            first_t, count = None, 0
            for out in inst.async_stream_infer(
                session_id=r+1, input_ids=input_ids, gen_config=gen_cfg,
                sequence_start=True, sequence_end=True,
            ):
                if out.status.value in (1, 2):
                    el = (time.perf_counter() - start) * 1000
                    if first_t is None:
                        first_t = el
                    count += len(out.token_ids)

            total_ms = (time.perf_counter() - start) * 1000
            ttft = first_t or 0
            decode_ms = total_ms - ttft
            decode_tps = (count / decode_ms * 1000) if decode_ms > 0 else 0
            prefill_tps = (len(input_ids) / ttft * 1000) if ttft > 0 else 0
            run_data.append({
                "ctx": len(input_ids), "ttft": ttft,
                "prefill_tps": prefill_tps, "decode_tps": decode_tps,
                "total_ms": total_ms, "tokens": count,
            })
            print(f"  [{ctx_label}] run={r+1}: ctx={len(input_ids)}, TTFT={ttft:.1f}ms, "
                  f"prefill={prefill_tps:.0f} t/s, decode={decode_tps:.0f} t/s, tokens={count}")

        # Average
        avg = {k: sum(d[k] for d in run_data)/len(run_data) for k in ["ttft", "prefill_tps", "decode_tps", "total_ms"]}
        results[ctx_label] = {"ctx_tokens": run_data[0]["ctx"], "avg": avg}
        print(f"  [{ctx_label}] AVG: TTFT={avg['ttft']:.1f}ms, "
              f"prefill={avg['prefill_tps']:.0f} t/s, decode={avg['decode_tps']:.0f} t/s")

    tm.close()
    return results

if __name__ == "__main__":
    print("TurboMind Performance Comparison")
    print(f"Model: Qwen3.5-9B, Output: {OUTPUT_TOKENS} tokens")

    r1 = run_test("Python Direct (Rust config)",
                  session_len=65536, max_batch_size=128, cache_block_seq_len=64)
    r2 = run_test("Python Direct (Python default config)",
                  session_len=8192, max_batch_size=32, cache_block_seq_len=64)

    results = {"python_rust_config": r1, "python_default_config": r2}
    with open("/mnt/data/lmdeploy/lmdeploy-rust-server/benchmark_python.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_python.json")
