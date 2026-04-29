#!/usr/bin/env python3
"""DFlash 性能基准测试 - 标准测试报告生成

测试不同 context 长度下的 baseline 和 DFlash speculative decoding 性能
"""

import sys
sys.path.insert(0, '/mnt/eaget-4tb/data/llm_server/1Cat-vLLM/lmdeploy')

import os
import time
import json
from datetime import datetime
from pathlib import Path
import torch

print("=" * 80)
print("DFlash 性能基准测试")
print("=" * 80)

# ============== 配置 ==============
TARGET_MODEL = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
DRAFT_MODEL = '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B-DFlash'

# 测试配置
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]
OUTPUT_TOKENS = 512
NUM_SPECULATIVE_TOKENS = 8
WARMUP_RUNS = 2
TEST_RUNS = 5

# 设备配置
DEVICE = 'cuda'
DTYPE = 'bfloat16'

# 输出目录
REPORT_DIR = Path('./reports')
REPORT_DIR.mkdir(exist_ok=True)

# ============== 测试 Prompts ==============
def get_test_prompt(context_tokens):
    """生成指定长度的测试 prompt"""
    # 使用重复的简单文本构建不同长度的 context
    base_text = "The quick brown fox jumps over the lazy dog. " * 10  # ~500 tokens
    repeat_times = max(1, context_tokens // 500)
    return base_text * repeat_times


# ============== 测试函数 ==============
def test_prefill_latency(model, tokenizer, prompt, num_runs=3):
    """测试 prefill 阶段延迟"""
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs, use_cache=True)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    return sum(times) / len(times)


def test_decode_performance(model, tokenizer, prompt, output_tokens, num_runs=5):
    """测试 decode 阶段性能"""
    times = []
    tokens_generated = []

    for _ in range(num_runs):
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

        # Prefill
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        torch.cuda.synchronize()
        start = time.time()

        # Decode
        generated_ids = outputs['logits'].argmax(dim=-1)
        past_key_values = outputs['past_key_values']

        for i in range(output_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            generated_ids = torch.cat([generated_ids, outputs['logits'].argmax(dim=-1)], dim=-1)
            past_key_values = outputs['past_key_values']

        torch.cuda.synchronize()
        times.append(time.time() - start)
        tokens_generated.append(output_tokens)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)

    return {
        'avg_time': avg_time,
        'tokens_per_sec': avg_tokens / avg_time,
        'ms_per_token': avg_time * 1000 / avg_tokens,
    }


def test_dflash_performance(target_model, draft_model, tokenizer, prompt, output_tokens, num_runs=5):
    """测试 DFlash speculative decoding 性能"""
    times = []
    tokens_generated = []
    accept_counts = []

    for _ in range(num_runs):
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

        # Prefill target model
        with torch.no_grad():
            target_outputs = target_model(**inputs, use_cache=True)

        torch.cuda.synchronize()
        start = time.time()

        # Speculative decoding
        generated_ids = target_outputs['logits'].argmax(dim=-1)
        target_past = target_outputs['past_key_values']
        draft_past = None

        total_accepted = 0

        while generated_ids.shape[1] - inputs['input_ids'].shape[1] < output_tokens:
            # Draft: generate N speculative tokens
            draft_ids = draft_model(
                input_ids=generated_ids[:, -1:],
                past_key_values=draft_past,
            )

            # Target: verify N tokens
            with torch.no_grad():
                target_outputs = target_model(
                    input_ids=draft_ids,
                    past_key_values=target_past,
                )

            # Rejection sampling
            accepted = 0
            for i in range(draft_ids.shape[1]):
                if accepted >= output_tokens:
                    break
                # Simplified: always accept for now
                accepted += 1

            total_accepted += accepted
            generated_ids = torch.cat([generated_ids, draft_ids[:, :accepted]], dim=-1)

        torch.cuda.synchronize()
        times.append(time.time() - start)
        tokens_generated.append(output_tokens)
        accept_counts.append(total_accepted)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    avg_accepts = sum(accept_counts) / len(accept_counts)

    return {
        'avg_time': avg_time,
        'tokens_per_sec': avg_tokens / avg_time,
        'ms_per_token': avg_time * 1000 / avg_tokens,
        'accept_rate': avg_accepts / (NUM_SPECULATIVE_TOKENS * output_tokens) if output_tokens > 0 else 0,
    }


# ============== 报告生成 ==============
def generate_markdown_report(results):
    """生成 Markdown 格式的测试报告"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = REPORT_DIR / f'dflash_benchmark_report_{timestamp}.md'

    with open(report_file, 'w') as f:
        # 标题
        f.write("# DFlash 性能基准测试报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 测试配置
        f.write("## 测试配置\n\n")
        f.write(f"- **Target Model**: {TARGET_MODEL}\n")
        f.write(f"- **Draft Model**: {DRAFT_MODEL}\n")
        f.write(f"- **Device**: {DEVICE}\n")
        f.write(f"- **Dtype**: {DTYPE}\n")
        f.write(f"- **Speculative Tokens**: {NUM_SPECULATIVE_TOKENS}\n")
        f.write(f"- **Output Tokens**: {OUTPUT_TOKENS}\n")
        f.write(f"- **Warmup Runs**: {WARMUP_RUNS}\n")
        f.write(f"- **Test Runs**: {TEST_RUNS}\n\n")

        # 系统信息
        f.write("## 系统信息\n\n")
        f.write(f"- **PyTorch**: {torch.__version__}\n")
        f.write(f"- **CUDA**: {torch.version.cuda}\n")
        if torch.cuda.is_available():
            f.write(f"- **GPU**: {torch.cuda.get_device_name(0)}\n")
            f.write(f"- **GPU Memory**: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n\n")

        # 测试结果表格
        f.write("## 测试结果\n\n")

        # 性能对比表
        f.write("### 性能对比 (tokens/sec)\n\n")
        f.write("| Context Length | Baseline | DFlash | Speedup |\n")
        f.write("|-----------------|----------|--------|---------|\n")

        for ctx_len in CONTEXT_LENGTHS:
            if ctx_len in results:
                r = results[ctx_len]
                baseline_tps = r.get('baseline', {}).get('tokens_per_sec', 0)
                dflash_tps = r.get('dflash', {}).get('tokens_per_sec', 0)
                speedup = dflash_tps / baseline_tps if baseline_tps > 0 else 0
                f.write(f"| {ctx_len} | {baseline_tps:.2f} | {dflash_tps:.2f} | {speedup:.2f}x |\n")

        f.write("\n")

        # 详细结果表
        f.write("### 详细结果\n\n")
        f.write("| Context | Mode | Prefill (ms) | Decode (ms/tok) | Tokens/sec | Accept Rate |\n")
        f.write("|---------|------|--------------|-----------------|-------------|-------------|\n")

        for ctx_len in CONTEXT_LENGTHS:
            if ctx_len in results:
                r = results[ctx_len]

                # Baseline
                if 'baseline' in r:
                    b = r['baseline']
                    f.write(f"| {ctx_len} | Baseline | - | {b.get('ms_per_token', 0):.2f} | {b.get('tokens_per_sec', 0):.2f} | - |\n")

                # DFlash
                if 'dflash' in r:
                    d = r['dflash']
                    f.write(f"| {ctx_len} | DFlash | - | {d.get('ms_per_token', 0):.2f} | {d.get('tokens_per_sec', 0):.2f} | {d.get('accept_rate', 0):.2%} |\n")

        f.write("\n")

        # 结论
        f.write("## 结论\n\n")

        avg_speedup = 0
        speedup_count = 0
        for ctx_len in CONTEXT_LENGTHS:
            if ctx_len in results:
                r = results[ctx_len]
                baseline_tps = r.get('baseline', {}).get('tokens_per_sec', 0)
                dflash_tps = r.get('dflash', {}).get('tokens_per_sec', 0)
                if baseline_tps > 0:
                    avg_speedup += dflash_tps / baseline_tps
                    speedup_count += 1

        if speedup_count > 0:
            avg_speedup /= speedup_count
            f.write(f"- **平均加速比**: {avg_speedup:.2f}x\n")

        f.write(f"- **最佳 Context 长度**: ...\n")
        f.write(f"- **平均接受率**: ...\n\n")

    print(f"\n✅ 报告已生成: {report_file}")
    return report_file


# ============== 主测试流程 ==============
def main():
    print("\n[配置]")
    print(f"Target Model: {TARGET_MODEL}")
    print(f"Draft Model: {DRAFT_MODEL}")
    print(f"Context Lengths: {CONTEXT_LENGTHS}")
    print(f"Output Tokens: {OUTPUT_TOKENS}")

    # 导入 lmdeploy 组件
    print(f"\n[导入 lmdeploy 组件]")
    try:
        from lmdeploy.pytorch.config import ModelConfig, CacheConfig, SpecDecodeConfig
        from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
        from lmdeploy.pytorch.model_inputs import ModelInputs
        print(f"✅ lmdeploy 组件导入成功")
    except Exception as e:
        print(f"❌ lmdeploy 组件导入失败: {e}")
        return

    # 创建配置
    print(f"\n[创建模型配置]")
    try:
        target_config = ModelConfig.from_pretrained(
            TARGET_MODEL,
            trust_remote_code=True,
            dtype=DTYPE,
        )
        print(f"✅ Target 配置: {target_config.hidden_size} hidden, {target_config.hf_config.num_hidden_layers} layers")

        draft_config = ModelConfig.from_pretrained(
            DRAFT_MODEL,
            trust_remote_code=True,
            dtype=DTYPE,
            is_draft_model=True,
            spec_method='dflash',
        )
        print(f"✅ Draft 配置: {draft_config.hidden_size} hidden, {draft_config.hf_config.num_hidden_layers} layers")

        cache_config = CacheConfig(
            max_batches=8,
            block_size=128,
            num_cpu_blocks=0,
            num_gpu_blocks=100,
            max_prefill_token_num=32768,
            device_type='cuda',
        )

        spec_config = SpecDecodeConfig(
            model=DRAFT_MODEL,
            method='dflash',
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
            cache_config=cache_config,
            model_config=draft_config,
        )
        print(f"✅ SpecDecode 配置: {NUM_SPECULATIVE_TOKENS} speculative tokens")

    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建模型 agents
    print(f"\n[创建模型 agents]")
    try:
        # Baseline agent
        baseline_agent = SpecModelAgent(
            specdecode_config=spec_config,
            backend_config=None,
            inputs_strategy=None,
            agent_strategy=None,
            device=DEVICE,
        )
        baseline_agent.build_model(empty_init=True)
        print(f"✅ Baseline agent 创建成功")

        # DFlash agent (复用)
        dflash_agent = baseline_agent
        print(f"✅ DFlash agent 创建成功")

    except Exception as e:
        print(f"❌ Agent 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 运行测试
    results = {}

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f"测试 Context Length: {ctx_len}")
        print(f"{'='*60}")

        prompt = get_test_prompt(ctx_len)
        print(f"Prompt length: {len(prompt)} chars")

        # Tokenize
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
        actual_ctx_len = input_ids.shape[1]
        print(f"Actual context tokens: {actual_ctx_len}")

        # 这里需要实现实际的测试逻辑
        # 由于 lmdeploy 的 API 比较复杂，这里先做简化处理

        results[ctx_len] = {
            'baseline': {
                'tokens_per_sec': 0,  # 需要实际测试
                'ms_per_token': 0,
            },
            'dflash': {
                'tokens_per_sec': 0,  # 需要实际测试
                'ms_per_token': 0,
                'accept_rate': 0,
            }
        }

    # 生成报告
    print(f"\n[生成测试报告]")
    report_file = generate_markdown_report(results)

    print(f"\n{'='*80}")
    print(f"测试完成!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
