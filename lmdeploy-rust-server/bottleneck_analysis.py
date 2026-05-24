#!/usr/bin/env python3
"""
Bottleneck analysis: measure each stage of the inference pipeline separately.

Stages measured:
1. Tokenization time
2. Engine forward call time (prefill + decode combined, synchronous)
3. Streaming token delivery time (via async callback)
4. Token decode time (per-token)

This reveals WHERE time is spent, not just end-to-end.
"""
import sys
sys.path.insert(0, '/mnt/data/lmdeploy')

import time
import asyncio

MODEL = '/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
OUTPUT_TOKENS = 64
RUNS = 3

# Use a single prompt length for focused analysis
PROMPT = "The quick brown fox jumps over the lazy dog. " * 80  # ~512 tokens


async def main():
    from lmdeploy.turbomind import TurboMind
    from lmdeploy.tokenizer import Tokenizer
    from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig

    print("=" * 70)
    print("TURBOMIND BOTTLENECK ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print(f"Runs: {RUNS}")

    # Load engine
    ec = TurbomindEngineConfig(
        session_len=8192,
        max_batch_size=32,
        cache_block_seq_len=64,
        tp=1,
        enable_prefix_caching=False,
    )
    print("\nLoading TurboMind...")
    tm = TurboMind(model_path=MODEL, engine_config=ec, trust_remote_code=True)
    tok = Tokenizer(MODEL, trust_remote_code=True)
    inst = tm.create_instance()

    # Encode prompt
    tok_start = time.perf_counter()
    input_ids = tok.encode(PROMPT)
    tok_ms = (time.perf_counter() - tok_start) * 1000
    ctx_len = len(input_ids)
    print(f"Prompt: {ctx_len} tokens after tokenization ({tok_ms:.2f}ms)")

    gen_cfg = GenerationConfig(max_new_tokens=OUTPUT_TOKENS, temperature=0.0)

    # ============ STAGE 1: Synchronous forward (no streaming) ============
    print("\n--- Stage 1: Synchronous forward() (blocking, all-in-one) ---")
    stage1_data = []
    # Stage 1: synchronous forward
    for r in range(RUNS):
        t0 = time.perf_counter()
        async for out in inst.async_stream_infer(
            session_id=r, input_ids=input_ids, gen_config=gen_cfg,
            sequence_start=True, sequence_end=True,
        ):
            pass  # consume all tokens
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        stage1_data.append(total_ms)
        print(f"  run={r+1}: {total_ms:.1f}ms")

    avg_stage1 = sum(stage1_data) / len(stage1_data)
    print(f"  AVG: {avg_stage1:.1f}ms")

    # ============ STAGE 2: Streaming with detailed timing ============
    print("\n--- Stage 2: Streaming async_stream_infer() (detailed timing) ---")
    stage2_data = []
    for r in range(RUNS):
        engine_start = time.perf_counter()

        first_intermediate_ms = None
        last_intermediate_ms = None
        total_tokens = 0
        token_timestamps = []
        finish_ms = 0.0

        async for out in inst.async_stream_infer(
            session_id=r + 100, input_ids=input_ids, gen_config=gen_cfg,
            sequence_start=True, sequence_end=True,
        ):
            elapsed_ms = (time.perf_counter() - engine_start) * 1000
            if out.status.value in (1, 2):  # intermediate
                if first_intermediate_ms is None:
                    first_intermediate_ms = elapsed_ms
                last_intermediate_ms = elapsed_ms
                total_tokens += len(out.token_ids)
                for _ in out.token_ids:
                    token_timestamps.append(elapsed_ms)
            elif out.status.value == 3:  # finish
                finish_ms = elapsed_ms

        total_ms = (time.perf_counter() - engine_start) * 1000
        ttft = first_intermediate_ms or 0

        stage2_data.append({
            "ttft_ms": ttft,
            "total_ms": total_ms,
            "finish_ms": finish_ms,
            "tokens": total_tokens,
            "token_timestamps": token_timestamps,
        })
        print(f"  run={r+1}: TTFT={ttft:.1f}ms, finish={finish_ms:.1f}ms, "
              f"total={total_ms:.1f}ms, tokens={total_tokens}")

    # Compute ITL (inter-token latency)
    avg_stage2 = {}
    for k in ["ttft_ms", "total_ms", "finish_ms"]:
        avg_stage2[k] = sum(d[k] for d in stage2_data) / len(stage2_data)

    # Inter-token latency: time between consecutive tokens
    if stage2_data[0]["token_timestamps"]:
        ts = stage2_data[0]["token_timestamps"]
        itls = [ts[i+1] - ts[i] for i in range(len(ts)-1)] if len(ts) > 1 else [0]
        avg_itl = sum(itls) / len(itls)
        avg_stage2["avg_itl_ms"] = avg_itl
    else:
        avg_stage2["avg_itl_ms"] = 0

    print(f"\n  AVG: TTFT={avg_stage2['ttft_ms']:.1f}ms, "
          f"finish={avg_stage2['finish_ms']:.1f}ms, "
          f"total={avg_stage2['total_ms']:.1f}ms")
    print(f"  Avg ITL: {avg_stage2['avg_itl_ms']:.3f}ms")

    decode_ms = avg_stage2['finish_ms'] - avg_stage2['ttft_ms']
    decode_tps = (OUTPUT_TOKENS / decode_ms * 1000) if decode_ms > 0 else 0
    prefill_tps = (ctx_len / avg_stage2['ttft_ms'] * 1000) if avg_stage2['ttft_ms'] > 0 else 0

    print(f"\n  Computed:")
    print(f"    Decode time: {decode_ms:.1f}ms")
    print(f"    Decode speed: {decode_tps:.1f} tok/s")
    print(f"    Prefill speed (TTFT-based): {prefill_tps:.0f} tok/s")
    print(f"    Overall throughput: {ctx_len / avg_stage2['total_ms'] * 1000:.0f} tok/s")

    # ============ STAGE 3: Engine-only forward (without tokenization) ============
    print("\n--- Stage 3: Engine-only (tokenization already done) ---")
    print(f"    (Same as Stage 2 since tokenization is done outside timer)")
    print(f"    Stage 2 already measures engine-only time")

    # ============ STAGE 4: What if we measure total end-to-end (including tokenization)? ============
    print("\n--- Stage 4: End-to-end (including tokenization) ---")
    stage4_data = []
    for r in range(RUNS):
        t0 = time.perf_counter()
        input_ids_tok = tok.encode(PROMPT)  # tokenization
        t_tok = time.perf_counter()
        async for out in inst.async_stream_infer(
            session_id=r + 200, input_ids=input_ids_tok, gen_config=gen_cfg,
            sequence_start=True, sequence_end=True,
        ):
            pass
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        tok_only_ms = (t_tok - t0) * 1000
        engine_only_ms = (t1 - t_tok) * 1000
        stage4_data.append({"total_ms": total_ms, "tok_ms": tok_only_ms, "engine_ms": engine_only_ms})
        print(f"  run={r+1}: total={total_ms:.1f}ms, tok={tok_only_ms:.2f}ms, engine={engine_only_ms:.1f}ms")

    avg_e2e = sum(d["total_ms"] for d in stage4_data) / len(stage4_data)
    avg_tok = sum(d["tok_ms"] for d in stage4_data) / len(stage4_data)
    avg_eng = sum(d["engine_ms"] for d in stage4_data) / len(stage4_data)

    print(f"\n  AVG: total={avg_e2e:.1f}ms, tokenization={avg_tok:.2f}ms, engine={avg_eng:.1f}ms")

    # ============ Summary ============
    print(f"\n{'='*70}")
    print("BOTTLENECK ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"  Context length: {ctx_len} tokens")
    print(f"  Output tokens: {OUTPUT_TOKENS}")
    print(f"")
    print(f"  Stage 1 (sync forward):          {avg_stage1:.1f}ms")
    print(f"  Stage 2 (streaming TTFT):        {avg_stage2['ttft_ms']:.1f}ms")
    print(f"  Stage 2 (streaming finish):      {avg_stage2['finish_ms']:.1f}ms")
    print(f"  Stage 2 (streaming total):       {avg_stage2['total_ms']:.1f}ms")
    print(f"  Stage 4 (e2e total):             {avg_e2e:.1f}ms")
    print(f"  Stage 4 (e2e tokenization):      {avg_tok:.2f}ms")
    print(f"  Stage 4 (e2e engine):            {avg_eng:.1f}ms")
    print(f"")
    print(f"  Tokenization overhead:           {avg_tok:.2f}ms ({avg_tok/avg_e2e*100:.1f}% of total)")
    print(f"  Engine compute time:             {avg_eng:.1f}ms ({avg_eng/avg_e2e*100:.1f}% of total)")
    print(f"")
    print(f"  Throughput (engine-only):        {ctx_len / avg_eng * 1000:.0f} tok/s")
    print(f"  Throughput (e2e):                {ctx_len / avg_e2e * 1000:.0f} tok/s")
    print(f"  Throughput (decode-only):        {decode_tps:.1f} tok/s")
    print(f"")
    print(f"  Prefill speed (TTFT-based):      {prefill_tps:.0f} tok/s")
    print(f"  Note: TTFT-based prefill conflates prefill + first decode step")
    print(f"        Actual prefill is faster than TTFT-based measurement suggests")

    tm.close()


asyncio.run(main())
