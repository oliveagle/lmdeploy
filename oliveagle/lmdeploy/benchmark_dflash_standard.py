#!/usr/bin/env python3
"""Standard DFlash Benchmark - Table Format Output

Tests baseline vs DFlash speculative decoding across different context lengths.
"""

import sys
import os
import time
import json
import multiprocessing
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

# Output directory
REPORT_DIR = Path('./reports')
REPORT_DIR.mkdir(exist_ok=True)

# Model paths
TARGET_MODEL = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B"
DRAFT_MODEL = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash"

# Test configuration
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192]  # context lengths to test (limited by GPU memory)
OUTPUT_TOKENS = 256  # tokens to generate
NUM_SPEC_TOKENS = 8  # speculative tokens

# Test prompts (different lengths) - use reusable base prompt
BASE_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20  # ~300 tokens

PROMPTS = {}
for ctx_len in CONTEXT_LENGTHS:
    # Calculate repeat times to reach target length
    repeat_times = max(1, ctx_len // 300)
    PROMPTS[ctx_len] = BASE_PROMPT * repeat_times

# Generation prompt
GEN_PROMPT = "Summarize the above text and explain the key concepts."


def count_tokens(text, tokenizer):
    """Count tokens in text."""
    return len(tokenizer.encode(text))


def run_single_test(mode, target_model, draft_model, prompt, max_tokens, num_spec_tokens, dtype='float16'):
    """Run a single test - must be called in a subprocess for memory isolation."""
    try:
        from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
        from transformers import AutoTokenizer

        # Get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        prompt_tokens = count_tokens(prompt, tokenizer)

        # Build pipeline
        if mode == 'dflash':
            from lmdeploy.messages import SpeculativeConfig
            from lmdeploy import PytorchEngineConfig
            spec_config = SpeculativeConfig(
                method='dflash',
                model=draft_model,
                num_speculative_tokens=num_spec_tokens,
            )
            backend_config = PytorchEngineConfig(
                dtype=dtype,
                cache_max_entry_count=0.5,
                eager_mode=True,  # Eager mode to avoid CUDA graph issues with multi-token
            )
        else:
            spec_config = None
            backend_config = TurbomindEngineConfig(
                dtype=dtype,
                cache_max_entry_count=0.7,
            )

        pipe = pipeline(
            target_model,
            speculative_config=spec_config,
            backend_config=backend_config,
            log_level='ERROR',
        )

        # Prefill
        start_prefill = time.time()
        _ = pipe(prompt, GenerationConfig(max_new_tokens=1))
        prefill_time = time.time() - start_prefill

        # Decode
        start_decode = time.time()
        output = pipe(
            prompt,
            GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            ),
        )
        decode_time = time.time() - start_decode

        num_tokens = len(output.token_ids)
        prefill_tps = prompt_tokens / prefill_time if prefill_time > 0 else 0
        decode_tps = num_tokens / decode_time if decode_time > 0 else 0

        return {
            'success': True,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'prefill_tps': prefill_tps,
            'decode_tps': decode_tps,
            'total_time': prefill_time + decode_time,
            'prompt_tokens': prompt_tokens,
            'output_tokens': num_tokens,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }


def _run_test_and_save(target_model, draft_model, prompt, max_tokens, num_spec_tokens, dtype, result_file):
    """Worker function to run a test and save results. Must be module-level for pickling."""
    mode = 'dflash' if draft_model is not None else 'baseline'
    result = run_single_test(mode, target_model, draft_model, prompt, max_tokens, num_spec_tokens, dtype)
    with open(result_file, 'w') as f:
        json.dump(result, f)


def _run_test_with_traceback(target_model, draft_model, prompt, max_tokens, num_spec_tokens, dtype, result_file):
    """Worker function with full traceback capture."""
    import traceback as tb
    mode = 'dflash' if draft_model is not None else 'baseline'
    try:
        result = run_single_test(mode, target_model, draft_model, prompt, max_tokens, num_spec_tokens, dtype)
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'traceback': tb.format_exc(),
        }
    with open(result_file, 'w') as f:
        json.dump(result, f)


def run_baseline(target_model, prompt, max_tokens, dtype='float16'):
    """Run baseline inference in a subprocess for GPU memory isolation."""
    import tempfile
    ctx = multiprocessing.get_context('spawn')
    result_file = tempfile.mktemp(suffix='.json')

    p = ctx.Process(target=_run_test_and_save, args=(target_model, None, prompt, max_tokens, 0, dtype, result_file))
    p.start()
    p.join()

    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)
    return {'success': False, 'error': 'Process failed to return result'}


def run_dflash(target_model, draft_model, prompt, max_tokens, num_spec_tokens, dtype='float16'):
    """Run DFlash speculative decoding in a subprocess for GPU memory isolation."""
    import tempfile
    ctx = multiprocessing.get_context('spawn')
    result_file = tempfile.mktemp(suffix='.json')

    p = ctx.Process(target=_run_test_with_traceback, args=(target_model, draft_model, prompt, max_tokens, num_spec_tokens, dtype, result_file))
    p.start()
    p.join()

    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)
    return {'success': False, 'error': 'Process failed to return result'}


def print_table(results):
    """Print results as Markdown table."""
    print("\n" + "=" * 120)
    print("DFlash Benchmark Results")
    print("=" * 120)
    print(f"Model: Qwen3.5-9B + DFlash Draft")
    print(f"Speculative Tokens: {NUM_SPEC_TOKENS}")
    print(f"Output Tokens: {OUTPUT_TOKENS}")
    print("=" * 120)

    # Header
    print("\n| Context | Mode      | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) | Speedup |")
    print("|---------|-----------|--------------|-------------|---------------|---------------|-----------|---------|")

    for ctx_len, result in results.items():
        ctx_label = f"{ctx_len//1024}K"

        # Baseline
        b = result['baseline']
        if b.get('success'):
            b_prefill_ms = b['prefill_time'] * 1000
            b_decode_ms = b['decode_time'] * 1000
            print(f"| {ctx_label} | Baseline  | {b_prefill_ms:10.1f} | {b_decode_ms:11.1f} | "
                  f"{b['prefill_tps']:11.1f} | {b['decode_tps']:11.1f} | "
                  f"{b['total_time']:8.2f} |    -    |")
        else:
            print(f"| {ctx_label} | Baseline  |     ERROR    | {b['error'][:30]:30s} | - | - | - | - |")

        # DFlash
        d = result['dflash']
        if d.get('success'):
            d_prefill_ms = d['prefill_time'] * 1000
            d_decode_ms = d['decode_time'] * 1000
            speedup = d['decode_tps'] / b['decode_tps'] if b.get('success') and b.get('decode_tps') else 0
            print(f"| {ctx_label} | DFlash    | {d_prefill_ms:10.1f} | {d_decode_ms:11.1f} | "
                  f"{d['prefill_tps']:11.1f} | {d['decode_tps']:11.1f} | "
                  f"{d['total_time']:8.2f} |  {speedup:5.2f}x |")
        else:
            print(f"| {ctx_label} | DFlash    |     ERROR    | {d['error'][:30]:30s} | - | - | - | - |")

    print("=" * 120)

    # Summary
    print("\nSummary Statistics:")
    baseline_decode = [r['baseline']['decode_tps'] for r in results.values() if r['baseline'].get('success')]
    dflash_decode = [r['dflash']['decode_tps'] for r in results.values() if r['dflash'].get('success')]

    if baseline_decode and dflash_decode:
        avg_baseline = sum(baseline_decode) / len(baseline_decode)
        avg_dflash = sum(dflash_decode) / len(dflash_decode)
        print(f"  Avg Baseline Decode:  {avg_baseline:.1f} tokens/s")
        print(f"  Avg DFlash Decode:     {avg_dflash:.1f} tokens/s")
        print(f"  Avg Decode Speedup:    {avg_dflash/avg_baseline:.2f}x")

    print("=" * 120)


def save_report(results, system_info=None):
    """Save results as Markdown report and JSON."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_file = REPORT_DIR / f'dflash_benchmark_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'target_model': TARGET_MODEL,
            'draft_model': DRAFT_MODEL,
            'spec_tokens': NUM_SPEC_TOKENS,
            'output_tokens': OUTPUT_TOKENS,
            'system_info': system_info,
            'results': results,
        }, f, indent=2)
    print(f"\n✅ JSON saved: {json_file}")

    # Save Markdown report
    md_file = REPORT_DIR / f'dflash_benchmark_{timestamp}.md'
    with open(md_file, 'w') as f:
        f.write("# DFlash 性能基准测试报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 测试配置\n\n")
        f.write("| 配置项 | 值 |\n")
        f.write("|--------|-----|\n")
        f.write(f"| **Target Model** | `{TARGET_MODEL}` |\n")
        f.write(f"| **Draft Model** | `{DRAFT_MODEL}` |\n")
        f.write(f"| **Speculative Tokens** | {NUM_SPEC_TOKENS} |\n")
        f.write(f"| **Output Tokens** | {OUTPUT_TOKENS} |\n\n")

        if system_info:
            f.write("## 系统信息\n\n")
            f.write("| 项目 | 值 |\n")
            f.write("|------|-----|\n")
            for key, value in system_info.items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")

        f.write("## 性能对比\n\n")
        f.write("### Decode 速度 (tokens/s)\n\n")
        f.write("| Context | Baseline | DFlash | Speedup |\n")
        f.write("|---------|----------|--------|---------|\n")

        for ctx_len, result in sorted(results.items()):
            ctx_label = f"{ctx_len//1024}K"
            b = result.get('baseline', {})
            d = result.get('dflash', {})
            baseline_tps = b.get('decode_tps', 0) if b.get('success') else 0
            dflash_tps = d.get('decode_tps', 0) if d.get('success') else 0
            speedup = dflash_tps / baseline_tps if baseline_tps > 0 else 0
            f.write(f"| {ctx_label} | {baseline_tps:.1f} | {dflash_tps:.1f} | {speedup:.2f}x |\n")

        f.write("\n")

        # Detailed table
        f.write("### 详细结果\n\n")
        f.write("| Context | Mode | Prefill (ms) | Decode (ms) | Prefill (t/s) | Decode (t/s) | Total (s) |\n")
        f.write("|---------|------|--------------|-------------|---------------|---------------|----------|\n")

        for ctx_len, result in sorted(results.items()):
            ctx_label = f"{ctx_len//1024}K"

            # Baseline
            b = result.get('baseline', {})
            if b.get('success'):
                f.write(f"| {ctx_label} | Baseline | {b['prefill_time']*1000:.1f} | {b['decode_time']*1000:.1f} | "
                       f"{b['prefill_tps']:.1f} | {b['decode_tps']:.1f} | {b['total_time']:.2f} |\n")

            # DFlash
            d = result.get('dflash', {})
            if d.get('success'):
                f.write(f"| {ctx_label} | DFlash | {d['prefill_time']*1000:.1f} | {d['decode_time']*1000:.1f} | "
                       f"{d['prefill_tps']:.1f} | {d['decode_tps']:.1f} | {d['total_time']:.2f} |\n")

        # Summary
        f.write("\n## 结论\n\n")
        baseline_decode = [r['baseline']['decode_tps'] for r in results.values() if r['baseline'].get('success')]
        dflash_decode = [r['dflash']['decode_tps'] for r in results.values() if r['dflash'].get('success')]

        if baseline_decode and dflash_decode:
            avg_baseline = sum(baseline_decode) / len(baseline_decode)
            avg_dflash = sum(dflash_decode) / len(dflash_decode)
            f.write(f"- **平均 Decode 速度**:\n")
            f.write(f"  - Baseline: {avg_baseline:.1f} tokens/s\n")
            f.write(f"  - DFlash: {avg_dflash:.1f} tokens/s\n")
            f.write(f"  - **加速比**: {avg_dflash/avg_baseline:.2f}x\n\n")

    print(f"✅ Markdown report saved: {md_file}")

    return json_file, md_file


def main():
    print("=" * 120)
    print("DFlash Standard Benchmark")
    print("=" * 120)
    print(f"Target Model: {TARGET_MODEL}")
    print(f"Draft Model:  {DRAFT_MODEL}")
    print(f"Spec Tokens:  {NUM_SPEC_TOKENS}")
    print(f"Output Tokens: {OUTPUT_TOKENS}")
    print("=" * 120)

    # Collect system info
    system_info = {}
    try:
        import torch
        system_info['PyTorch'] = torch.__version__
        system_info['CUDA'] = torch.version.cuda if torch.cuda.is_available() else 'N/A'
        if torch.cuda.is_available():
            system_info['GPU'] = torch.cuda.get_device_name(0)
            system_info['GPU Memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    except ImportError:
        pass

    for k, v in system_info.items():
        print(f"  {k}: {v}")

    results = {}
    tokenizer = None

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'='*120}")
        print(f"Testing Context Length: {ctx_len} ({ctx_len//1024}K)")
        print(f"{'='*120}")

        prompt = PROMPTS[ctx_len]

        # Estimate token count
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
        actual_tokens = len(tokenizer.encode(prompt))
        print(f"Prompt actual tokens: {actual_tokens}")

        result = {}

        # Baseline
        print("\n[1/2] Running Baseline...")
        baseline = run_baseline(TARGET_MODEL, prompt, OUTPUT_TOKENS, dtype='float16')
        result['baseline'] = baseline
        if baseline.get('success'):
            print(f"  ✓ Prefill: {baseline['prefill_time']*1000:.1f}ms ({baseline['prefill_tps']:.1f} t/s)")
            print(f"  ✓ Decode:  {baseline['decode_time']*1000:.1f}ms ({baseline['decode_tps']:.1f} t/s)")
        else:
            print(f"  ✗ Error: {baseline['error']}")

        # DFlash
        print("\n[2/2] Running DFlash...")
        dflash = run_dflash(TARGET_MODEL, DRAFT_MODEL, prompt, OUTPUT_TOKENS, NUM_SPEC_TOKENS, dtype='float16')
        result['dflash'] = dflash
        if dflash.get('success'):
            print(f"  ✓ Prefill: {dflash['prefill_time']*1000:.1f}ms ({dflash['prefill_tps']:.1f} t/s)")
            print(f"  ✓ Decode:  {dflash['decode_time']*1000:.1f}ms ({dflash['decode_tps']:.1f} t/s)")
            if baseline.get('success'):
                speedup = dflash['decode_tps'] / baseline['decode_tps']
                print(f"  ✓ Speedup: {speedup:.2f}x")
        else:
            print(f"  ✗ Error: {dflash['error']}")

            # Save detailed error for debugging
            error_file = REPORT_DIR / f'dflash_error_{ctx_len}.txt'
            with open(error_file, 'w') as f:
                f.write(f"Error at context length {ctx_len}:\n")
                f.write(f"{dflash['error']}\n")
                f.write("\nNote: This error needs investigation. Check model implementation.\n")
            print(f"  📝 Error details saved to: {error_file}")

        results[ctx_len] = result

    # Print table
    print_table(results)

    # Save report
    save_report(results, system_info)

    return 0


if __name__ == '__main__':
    sys.exit(main())
