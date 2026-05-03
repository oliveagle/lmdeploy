#!/usr/bin/env python3
"""Performance test script for AWQ MoE implementation.

Tests context lengths: 4k, 8k, 16k
Tests stages: prefill, decode
Metrics: latency, throughput, accept rate
"""

import argparse
import json
import sys
import time
import torch
import numpy as np


def get_device_info():
    """Get device information."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(device)
        info = {
            'device': 'cuda',
            'name': torch.cuda.get_device_name(device),
            'total_memory_gb': props.total_memory / (1024**3),
            'compute_capability': f'{props.major}.{props.minor}',
        }
    else:
        info = {
            'device': 'cpu',
            'name': 'CPU',
            'total_memory_gb': None,
            'compute_capability': None,
        }
    return info


def create_test_inputs(batch_size, seq_len, hidden_dim, num_experts, top_k, device):
    """Create test inputs for MoE."""
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    topk_weights = torch.rand(batch_size, seq_len, top_k, dtype=torch.float16, device=device)
    # Normalize weights
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (batch_size, seq_len, top_k), device=device)
    return hidden_states, topk_weights, topk_ids


def create_awq_moe_model(hidden_dim, ffn_dim, num_experts, top_k, w_bit, group_size, device):
    """Create AWQ MoE model for testing."""
    from lmdeploy.pytorch.nn.moe.awq import FusedMoEAWQ

    moe = FusedMoEAWQ(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        num_experts=num_experts,
        top_k=top_k,
        w_bit=w_bit,
        group_size=group_size,
        bias=False,
        renormalize=True,
        dtype=torch.float16,
        device=device,
        all_reduce=False,
        layer_idx=0,
        act_func=None,
    )

    # Initialize weights with random data
    with torch.no_grad():
        # qweight: (num_experts, in_features, quant_out_feats)
        moe.gate_up.qweight.data = torch.randint(
            0, 256, moe.gate_up.qweight.shape, dtype=torch.int32, device=device
        )
        moe.down.qweight.data = torch.randint(
            0, 256, moe.down.qweight.shape, dtype=torch.int32, device=device
        )

        # scales: (num_experts, grouped_in_feats, out_features)
        moe.gate_up.scales.data = torch.rand(
            moe.gate_up.scales.shape, dtype=torch.float16, device=device
        )
        moe.down.scales.data = torch.rand(
            moe.down.scales.shape, dtype=torch.float16, device=device
        )

        # qzeros: (num_experts, grouped_in_feats, quant_out_feats)
        moe.gate_up.qzeros.data = torch.randint(
            0, 256, moe.gate_up.qzeros.shape, dtype=torch.int32, device=device
        )
        moe.down.qzeros.data = torch.randint(
            0, 256, moe.down.qzeros.shape, dtype=torch.int32, device=device
        )

    return moe


def warmup_model(moe, hidden_states, topk_weights, topk_ids, num_iterations=3):
    """Warmup the model."""
    for i in range(num_iterations):
        with torch.no_grad():
            _ = moe(hidden_states, topk_weights, topk_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None


def measure_prefill_latency(moe, hidden_states, topk_weights, topk_ids, num_iterations=10):
    """Measure prefill latency."""
    latencies = []

    # Warmup
    warmup_model(moe, hidden_states, topk_weights, topk_ids, num_iterations=3)

    print(f"Measuring prefill latency ({num_iterations} iterations)...")
    for i in range(num_iterations):
        # Create new random inputs each iteration to avoid any caching
        batch_size, seq_len, hidden_dim = hidden_states.shape
        new_hidden = torch.randn_like(hidden_states)
        new_topk_weights = torch.rand_like(topk_weights)
        new_topk_weights = new_topk_weights / new_topk_weights.sum(dim=-1, keepdim=True)
        new_topk_ids = torch.randint(0, topk_ids.max().item() + 1, topk_ids.shape, device=topk_ids.device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        with torch.no_grad():
            output = moe(new_hidden, new_topk_weights, new_topk_ids)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    return output, latencies


def measure_decode_latency(moe, hidden_states, topk_weights, topk_ids, num_iterations=100):
    """Measure decode latency (single token)."""
    latencies = []

    # Warmup
    warmup_model(moe, hidden_states[:, :1, :], topk_weights[:, :1, :], topk_ids[:, :1, :], num_iterations=3)

    print(f"Measuring decode latency ({num_iterations} iterations)...")
    for i in range(num_iterations):
        # Create new random inputs each iteration to avoid any caching
        batch_size, seq_len, hidden_dim = hidden_states.shape
        new_hidden = torch.randn(batch_size, 1, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        new_topk_weights = torch.rand(batch_size, 1, topk_weights.size(-1),
                                     dtype=topk_weights.dtype, device=topk_weights.device)
        new_topk_weights = new_topk_weights / new_topk_weights.sum(dim=-1, keepdim=True)
        new_topk_ids = torch.randint(0, topk_ids.max().item() + 1,
                                    (batch_size, 1, topk_ids.size(-1)),
                                    device=topk_ids.device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        with torch.no_grad():
            output = moe(new_hidden, new_topk_weights, new_topk_ids)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    return output, latencies


def calculate_accept_rate(topk_ids, num_experts):
    """Calculate expert utilization rate."""
    # Count unique experts used
    unique_experts = torch.unique(topk_ids).numel()
    accept_rate = unique_experts / num_experts
    return float(accept_rate)


def run_performance_test(
    batch_size=1,
    context_lengths=[4096, 8192, 16384],
    hidden_dim=5120,  # Qwen3.6-35B hidden size
    ffn_dim=13824,    # Qwen3.6-35B FFN size
    num_experts=64,   # Qwen3.6-35B-A3B experts
    top_k=8,          # Qwen3.6-35B-A3B top_k
    w_bit=4,
    group_size=128,
    num_iterations_prefill=10,
    num_iterations_decode=100,
):
    """Run performance test for AWQ MoE."""
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires a GPU.")
        return None

    device = torch.device('cuda')
    device_info = get_device_info()

    print("=" * 80)
    print("AWQ MoE Performance Test - Stage 1")
    print("=" * 80)
    print(f"Device: {device_info['name']} ({device_info['compute_capability']})")
    print(f"Total Memory: {device_info['total_memory_gb']:.2f} GB")
    print(f"Model Config:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  ffn_dim: {ffn_dim}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    print(f"  w_bit: {w_bit}")
    print(f"  group_size: {group_size}")
    print(f"  batch_size: {batch_size}")
    print("=" * 80)

    # Get initial memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024**3)

    # Create model
    print("\nCreating AWQ MoE model...")
    moe = create_awq_moe_model(hidden_dim, ffn_dim, num_experts, top_k, w_bit, group_size, device)

    model_memory = torch.cuda.memory_allocated() / (1024**3)
    print(f"Model memory: {model_memory:.2f} GB")

    results = {
        'device_info': device_info,
        'model_config': {
            'hidden_dim': hidden_dim,
            'ffn_dim': ffn_dim,
            'num_experts': num_experts,
            'top_k': top_k,
            'w_bit': w_bit,
            'group_size': group_size,
            'batch_size': batch_size,
        },
        'model_memory_gb': model_memory,
        'context_lengths': [],
    }

    # Test each context length
    for seq_len in context_lengths:
        print(f"\n{'=' * 80}")
        print(f"Testing context length: {seq_len}")
        print(f"{'=' * 80}")

        context_result = {
            'seq_len': seq_len,
            'prefill': {},
            'decode': {},
        }

        # Create test inputs
        hidden_states, topk_weights, topk_ids = create_test_inputs(
            batch_size, seq_len, hidden_dim, num_experts, top_k, device
        )

        # Measure prefill latency
        print(f"\n--- Prefill (batch_size={batch_size}, seq_len={seq_len}) ---")
        try:
            output, prefill_latencies = measure_prefill_latency(
                moe, hidden_states, topk_weights, topk_ids, num_iterations_prefill
            )

            prefill_mean = np.mean(prefill_latencies)
            prefill_std = np.std(prefill_latencies)
            prefill_p50 = np.percentile(prefill_latencies, 50)
            prefill_p95 = np.percentile(prefill_latencies, 95)
            prefill_p99 = np.percentile(prefill_latencies, 99)

            # Calculate throughput
            tokens_per_second = (batch_size * seq_len) / (prefill_mean / 1000)

            # Calculate accept rate
            accept_rate = calculate_accept_rate(topk_ids, num_experts)

            context_result['prefill'] = {
                'latency_ms': {
                    'mean': float(prefill_mean),
                    'std': float(prefill_std),
                    'p50': float(prefill_p50),
                    'p95': float(prefill_p95),
                    'p99': float(prefill_p99),
                },
                'throughput_tokens_per_sec': float(tokens_per_second),
                'accept_rate': float(accept_rate),
                'iterations': num_iterations_prefill,
            }

            print(f"Latency (ms):")
            print(f"  Mean: {prefill_mean:.2f}")
            print(f"  Std:  {prefill_std:.2f}")
            print(f"  P50:  {prefill_p50:.2f}")
            print(f"  P95:  {prefill_p95:.2f}")
            print(f"  P99:  {prefill_p99:.2f}")
            print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
            print(f"Accept Rate: {accept_rate:.2%}")

        except Exception as e:
            print(f"Prefill test failed: {e}")
            import traceback
            traceback.print_exc()
            context_result['prefill'] = {'error': str(e)}

        # Measure decode latency (single token)
        print(f"\n--- Decode (batch_size={batch_size}, seq_len=1) ---")
        try:
            output, decode_latencies = measure_decode_latency(
                moe, hidden_states, topk_weights, topk_ids, num_iterations_decode
            )

            decode_mean = np.mean(decode_latencies)
            decode_std = np.std(decode_latencies)
            decode_p50 = np.percentile(decode_latencies, 50)
            decode_p95 = np.percentile(decode_latencies, 95)
            decode_p99 = np.percentile(decode_latencies, 99)

            # Calculate throughput
            tokens_per_second = batch_size / (decode_mean / 1000)

            context_result['decode'] = {
                'latency_ms': {
                    'mean': float(decode_mean),
                    'std': float(decode_std),
                    'p50': float(decode_p50),
                    'p95': float(decode_p95),
                    'p99': float(decode_p99),
                },
                'throughput_tokens_per_sec': float(tokens_per_second),
                'iterations': num_iterations_decode,
            }

            print(f"Latency (ms):")
            print(f"  Mean: {decode_mean:.2f}")
            print(f"  Std:  {decode_std:.2f}")
            print(f"  P50:  {decode_p50:.2f}")
            print(f"  P95:  {decode_p95:.2f}")
            print(f"  P99:  {decode_p99:.2f}")
            print(f"Throughput: {tokens_per_second:.2f} tokens/sec")

        except Exception as e:
            print(f"Decode test failed: {e}")
            import traceback
            traceback.print_exc()
            context_result['decode'] = {'error': str(e)}

        results['context_lengths'].append(context_result)

        # Clean up
        del hidden_states, topk_weights, topk_ids, output
        torch.cuda.empty_cache()

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    results['peak_memory_gb'] = peak_memory

    print(f"\n{'=' * 80}")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    print(f"{'=' * 80}")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='AWQ MoE Performance Test')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--context-lengths', type=int, nargs='+', default=[4096, 8192, 16384],
                        help='Context lengths to test')
    parser.add_argument('--hidden-dim', type=int, default=5120, help='Hidden dimension')
    parser.add_argument('--ffn-dim', type=int, default=13824, help='FFN dimension')
    parser.add_argument('--num-experts', type=int, default=64, help='Number of experts')
    parser.add_argument('--top-k', type=int, default=8, help='Top-k experts')
    parser.add_argument('--w-bit', type=int, default=4, help='Weight bit width')
    parser.add_argument('--group-size', type=int, default=128, help='Group size for quantization')
    parser.add_argument('--num-iterations-prefill', type=int, default=10, help='Number of prefill iterations')
    parser.add_argument('--num-iterations-decode', type=int, default=100, help='Number of decode iterations')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    # Run performance test
    results = run_performance_test(
        batch_size=args.batch_size,
        context_lengths=args.context_lengths,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        w_bit=args.w_bit,
        group_size=args.group_size,
        num_iterations_prefill=args.num_iterations_prefill,
        num_iterations_decode=args.num_iterations_decode,
    )

    if results is None:
        return 1

    # Save results to JSON if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")

    for ctx in results['context_lengths']:
        if 'prefill' in ctx and 'latency_ms' in ctx['prefill']:
            print(f"\nContext Length: {ctx['seq_len']}")
            print(f"  Prefill Latency: {ctx['prefill']['latency_ms']['mean']:.2f} ms")
            print(f"  Prefill Throughput: {ctx['prefill']['throughput_tokens_per_sec']:.2f} tokens/sec")
            print(f"  Accept Rate: {ctx['prefill']['accept_rate']:.2%}")

        if 'decode' in ctx and 'latency_ms' in ctx['decode']:
            print(f"  Decode Latency: {ctx['decode']['latency_ms']['mean']:.2f} ms")
            print(f"  Decode Throughput: {ctx['decode']['throughput_tokens_per_sec']:.2f} tokens/sec")

    return 0


if __name__ == '__main__':
    sys.exit(main())
