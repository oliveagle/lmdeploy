#!/usr/bin/env python3
"""
1Cat-LMDeploy Context Scaling Benchmark (v3)
使用同步 infer() API，每次用不同 prompt 避免 prefix cache 命中
"""
import os
import time
import json
import random
import statistics
from datetime import datetime
from pathlib import Path

MODEL = '/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ'

TEST_CONTEXTS = [
    4096,      # 4K
    8192,      # 8K
    16384,     # 16K
    32768,     # 32K
    65536,     # 64K
    131072,    # 128K
    262144,    # 256K
    524288,    # 512K
    1048576,   # 1M
]

OUTPUT_DIR = Path('/mnt/eaget-4tb/llama_cpp/eval/results/stage1')

# 预生成真实感文本段落池（中英文混合），避免 prefix cache 命中
TEXT_POOL = []
_pool_seeded = False


def _ensure_pool():
    """预生成文本段落池，约 100 个段落，每段约 200-400 tokens"""
    global TEXT_POOL, _pool_seeded
    if _pool_seeded:
        return
    _pool_seeded = True

    paragraphs = [
        "人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟人类智能行为的系统。从早期的专家系统到如今的深度学习模型，AI技术经历了多次范式转换。当前的大语言模型通过海量数据训练，展现出了前所未有的语言理解和生成能力。这些模型不仅能够进行自然语言对话，还能完成编程、数学推理、创意写作等复杂任务。",
        "The development of machine learning has fundamentally transformed how we approach complex problems in computer science. Deep neural networks, particularly transformer architectures, have enabled breakthroughs in natural language processing, computer vision, and reinforcement learning. The scaling laws observed in recent years suggest that larger models trained on more data consistently achieve better performance across diverse benchmarks.",
        "深度学习框架的发展使得研究人员能够更高效地进行模型训练和实验。PyTorch和TensorFlow等框架提供了自动微分、GPU加速和分布式训练等关键功能。近年来，JAX等新兴框架也受到了广泛关注，它们通过函数式编程范式提供了更灵活的计算图构建方式。",
        "Quantum computing represents a paradigm shift in computational capability. Unlike classical bits that exist in states 0 or 1, quantum bits (qubits) can exist in superposition, enabling quantum computers to process multiple states simultaneously. This property, combined with entanglement and quantum interference, allows quantum algorithms to solve certain problems exponentially faster than their classical counterparts.",
        "自然语言处理领域在近年来取得了显著的进展。从最初的基于规则的方法，到统计机器学习模型，再到如今的预训练语言模型，NLP技术的每一次演进都带来了性能的大幅提升。当前最先进的模型能够理解和生成多种语言的文本，并在翻译、摘要、问答等任务上达到了接近人类水平的性能。",
        "The architecture of modern data centers has evolved significantly to meet the demands of cloud computing and AI workloads. High-speed interconnects like NVLink and InfiniBand enable efficient communication between GPUs, while specialized hardware accelerators such as TPUs and AI-specific ASICs provide massive computational throughput for training and inference tasks.",
        "计算机视觉技术在自动驾驶、医学影像分析、安防监控等领域有着广泛的应用。卷积神经网络的发明是该领域的里程碑事件，它通过局部感受野和权值共享机制，大幅减少了模型参数数量，同时保持了出色的特征提取能力。近年来，Vision Transformer的出现又为计算机视觉带来了新的研究范式。",
        "Reinforcement learning has demonstrated remarkable capabilities in game playing, robotics, and decision-making systems. Deep Q-Networks and Proximal Policy Optimization have become standard algorithms in the field. The combination of reinforcement learning with large language models has opened new avenues for AI alignment and instruction following.",
        "分布式系统设计需要考虑一致性、可用性和分区容错性这三个核心要素。根据CAP定理，在发生网络分区时，系统只能在一致性和可用性之间选择其一。现代分布式数据库系统通常采用最终一致性模型来提供高可用性，同时通过各种共识协议（如Raft和Paxos）来保证数据的正确性。",
        "The field of robotics has seen tremendous advances through the integration of AI techniques. Robot learning, manipulation, and navigation have all benefited from deep learning approaches. Sim-to-real transfer techniques allow models trained in simulation to be deployed on physical robots, significantly reducing the cost and time required for training in real-world environments.",
        "密码学是信息安全的基础，现代密码学体系包括对称加密、非对称加密、哈希函数和数字签名等核心组件。RSA和椭圆曲线密码算法是目前最广泛使用的公钥加密方案，而AES则是对称加密的事实标准。量子计算的崛起对现有密码体系构成了潜在威胁，后量子密码学的研究因此变得日益重要。",
        "Cloud computing has revolutionized how organizations manage their IT infrastructure. Infrastructure as a Service, Platform as a Service, and Software as a Service provide different levels of abstraction for deploying applications. Container orchestration platforms like Kubernetes have become the de facto standard for managing microservices at scale.",
        "边缘计算将计算资源从中心化的云数据中心推向网络边缘，使得数据处理更加接近数据源。这种架构能够显著降低延迟，提高响应速度，并减少网络带宽消耗。在物联网、自动驾驶和工业自动化等场景中，边缘计算发挥着越来越重要的作用。",
        "The study of algorithms and data structures forms the foundation of computer science. From sorting algorithms like quicksort and mergesort to graph algorithms like Dijkstra's shortest path, these fundamental tools enable efficient problem-solving. Advanced data structures such as B-trees, hash tables, and skip lists are essential building blocks for database systems and search engines.",
        "数据库系统是现代软件架构的核心组件。关系型数据库通过SQL语言提供了强大的数据查询和管理能力，而NoSQL数据库则针对特定场景（如文档存储、图数据库、时序数据）提供了更灵活的方案。NewSQL数据库试图兼顾关系型数据库的ACID特性和NoSQL数据库的水平扩展能力。",
        "Functional programming paradigms have gained popularity in recent years, influencing the design of mainstream languages. Concepts such as immutability, pure functions, and higher-order functions promote code that is easier to reason about and test. Languages like Haskell, Erlang, and Scala demonstrate different approaches to functional programming.",
        "软件工程的最佳实践包括版本控制、持续集成、自动化测试和代码审查等。敏捷开发方法论强调迭代开发和快速反馈，DevOps文化则促进了开发和运维团队的紧密协作。微服务架构将大型应用拆分为小型、独立部署的服务，提高了系统的可维护性和可扩展性。",
        "Computer graphics and rendering techniques have advanced dramatically with GPU computing. Ray tracing, once considered too computationally expensive for real-time applications, can now be performed in real-time thanks to dedicated hardware accelerators. Neural rendering techniques combine traditional graphics pipelines with machine learning to achieve photorealistic results.",
        "网络安全是数字化转型过程中不可忽视的重要领域。零信任安全模型取代了传统的边界防御策略，强调对所有访问请求进行严格的身份验证和授权。威胁检测和响应系统利用机器学习算法来识别异常行为模式，从而及时发现和阻止潜在的安全威胁。",
        "The Internet of Things connects billions of devices worldwide, generating massive amounts of data that require sophisticated processing and analysis. Edge computing and fog computing architectures distribute processing closer to data sources. Security and privacy concerns remain significant challenges in IoT deployments, requiring robust authentication and encryption mechanisms.",
    ]
    TEXT_POOL = paragraphs


def generate_prompt_text(token_target: int, seed: int = None) -> str:
    """从文本池随机拼接，生成接近目标 token 数的 prompt"""
    _ensure_pool()
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # 每个段落约 200-400 tokens（中英文混合），取平均 300
    avg_tokens_per_para = 300
    num_paras = max(1, token_target // avg_tokens_per_para + 2)

    # 随机选取段落，允许重复
    selected = [rng.choice(TEXT_POOL) for _ in range(num_paras)]
    text = "\n\n".join(selected)

    # 截取到大致目标长度（按字符估算，1 token ≈ 1.5 中文字 / 4 英文字符）
    char_target = token_target * 3
    if len(text) > char_target:
        text = text[:char_target]

    # 添加唯一前缀确保不会命中 prefix cache
    unique_prefix = f"[bench-{rng.randint(100000,999999)}] "
    return unique_prefix + text


def benchmark_context(pipe, context_len: int, num_warmup: int = 1, num_tests: int = 3):
    """对一个 context 长度进行 prefill 和 decode 测试"""
    from lmdeploy import GenerationConfig

    print(f"\n{'='*70}")
    print(f"测试 Context: {context_len:,} tokens ({context_len//1024}K)")
    print(f"{'='*70}")

    results = {
        'context_len': context_len,
        'prefill': None,
        'decode': None,
    }

    # === Prefill 测试 ===
    print(f"\n[Prefill] 性能测试...")
    prefill_len = min(8192, context_len // 2)
    if prefill_len < 2048:
        prefill_len = min(2048, context_len - 128)

    gen_config_prefill = GenerationConfig(max_new_tokens=1, temperature=0.0)
    print(f"  Prompt 长度：~{prefill_len} tokens")

    # 热身
    for i in range(num_warmup):
        try:
            warmup_text = generate_prompt_text(prefill_len, seed=context_len * 1000 + i)
            resp = pipe.infer(warmup_text, gen_config=gen_config_prefill)
            print(f"  热身 {i+1}/{num_warmup} ✓")
        except Exception as e:
            print(f"  热身 {i+1}/{num_warmup} ✗: {e}")
            return results

    # 正式测试 - 每次不同 seed 避免 prefix cache
    prefill_times = []
    for i in range(num_tests):
        try:
            test_seed = random.randint(0, 100000000)
            test_text = generate_prompt_text(prefill_len, seed=test_seed)
            start = time.perf_counter()
            resp = pipe.infer(test_text, gen_config=gen_config_prefill)
            elapsed = time.perf_counter() - start
            prefill_times.append(elapsed)
            throughput = prefill_len / elapsed
            print(f"  测试 {i+1}/{num_tests}: {elapsed:.4f}s, {throughput:.2f} tok/s")
        except Exception as e:
            print(f"  测试 {i+1}/{num_tests} ✗: {e}")

    if prefill_times:
        avg_time = statistics.mean(prefill_times)
        std_time = statistics.stdev(prefill_times) if len(prefill_times) > 1 else 0
        avg_throughput = prefill_len / avg_time
        max_throughput = prefill_len / min(prefill_times)
        results['prefill'] = {
            'prompt_len': prefill_len,
            'avg_time': round(avg_time, 4),
            'std_time': round(std_time, 4),
            'avg_tok_s': round(avg_throughput, 2),
            'max_tok_s': round(max_throughput, 2),
        }
        print(f"  汇总：{avg_throughput:.2f} tok/s (avg), {max_throughput:.2f} tok/s (max)")

    # === Decode 测试 ===
    print(f"\n[Decode] 性能测试...")
    decode_len = 128
    gen_config_decode = GenerationConfig(max_new_tokens=decode_len, temperature=0.0)

    # 热身 + 正式测试合并，每次不同 prompt
    decode_times = []
    gen_tokens = []
    total_runs = num_warmup + num_tests

    for i in range(total_runs):
        try:
            decode_text = generate_prompt_text(32, seed=random.randint(0, 100000000))
            start = time.perf_counter()
            resp = pipe.infer(decode_text, gen_config=gen_config_decode)
            elapsed = time.perf_counter() - start

            if i < num_warmup:
                print(f"  热身 {i+1}/{num_warmup} ✓ ({elapsed:.4f}s)")
            else:
                actual_tokens = len(resp.token_ids) if hasattr(resp, 'token_ids') else decode_len
                decode_times.append(elapsed)
                gen_tokens.append(actual_tokens)
                throughput = actual_tokens / elapsed
                idx = i - num_warmup + 1
                print(f"  测试 {idx}/{num_tests}: {elapsed:.4f}s, {throughput:.2f} tok/s ({actual_tokens} tokens)")
        except Exception as e:
            idx = i - num_warmup + 1 if i >= num_warmup else i + 1
            print(f"  ✗: {e}")

    if decode_times and gen_tokens:
        avg_time = statistics.mean(decode_times)
        std_time = statistics.stdev(decode_times) if len(decode_times) > 1 else 0
        avg_tokens = statistics.mean(gen_tokens)
        avg_throughput = avg_tokens / avg_time
        max_throughput = max(gen_tokens) / min(decode_times)
        results['decode'] = {
            'gen_len': int(avg_tokens),
            'avg_time': round(avg_time, 4),
            'std_time': round(std_time, 4),
            'avg_tok_s': round(avg_throughput, 2),
            'max_tok_s': round(max_throughput, 2),
        }
        print(f"  汇总：{avg_throughput:.2f} tok/s (avg), {max_throughput:.2f} tok/s (max)")

    return results


def main():
    from lmdeploy import pipeline, TurbomindEngineConfig

    print("="*70)
    print("1Cat-LMDeploy Context Scaling Benchmark (v3)")
    print("="*70)
    print(f"模型：{MODEL}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试序列：{' -> '.join([f'{c//1024}K' for c in TEST_CONTEXTS])}")

    # 一次性加载模型，使用最大 session_len
    max_session = TEST_CONTEXTS[-1]
    print(f"\n加载模型 (session_len={max_session:,}, kv_cache_dtype=int4)...")

    try:
        start = time.time()
        backend_config = TurbomindEngineConfig(
            session_len=max_session,
            max_batch_size=1,  # 单请求，全部 KV cache 给一个 session
            cache_max_entry_count=0.8,
            kv_cache_dtype='int4',
        )
        pipe = pipeline(
            model_path=MODEL,
            backend_config=backend_config,
            tp=1,
        )
        load_time = time.time() - start
        print(f"✓ Load OK ({load_time:.1f}s)")
    except Exception as e:
        print(f"✗ Load Failed: {str(e)[:300]}")
        return

    all_results = {
        'model': 'Qwen3.6-35B-A3B',
        'backend': 'lmdeploy-turbomind',
        'version': '0.12.3',
        'timestamp': datetime.now().isoformat(),
        'load_time': round(load_time, 1),
        'config': {
            'cache_max_entry_count': 0.8,
            'kv_cache_dtype': 'int4',
            'session_len': max_session,
        },
        'tests': []
    }

    for context_len in TEST_CONTEXTS:
        result = benchmark_context(pipe, context_len)
        all_results['tests'].append(result)

        if result['prefill'] is None and result['decode'] is None:
            print(f"\n⚠️  {context_len:,} ({context_len//1024}K) 测试全部失败")
            break

    # 清理
    try:
        pipe.close()
    except:
        pass

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = OUTPUT_DIR / f"qwen3_6_35b_a3b_lmdeploy_int4_4k_up_{timestamp}.jsonc"

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存：{json_file}")

    # 汇总表格
    print("\n" + "="*70)
    print("汇总结果")
    print("="*70)
    print()
    print(f"{'Context':<12} {'Prefill (tok/s)':<20} {'Decode (tok/s)':<20}")
    print(f"{'':<12} {'avg':<10} {'max':<10} {'avg':<10} {'max':<10}")
    print("-"*60)

    for test in all_results['tests']:
        ctx_str = f"{test['context_len']//1024}K"
        p_avg = f"{test['prefill']['avg_tok_s']:.1f}" if test['prefill'] else "-"
        p_max = f"{test['prefill']['max_tok_s']:.1f}" if test['prefill'] else "-"
        d_avg = f"{test['decode']['avg_tok_s']:.1f}" if test['decode'] else "-"
        d_max = f"{test['decode']['max_tok_s']:.1f}" if test['decode'] else "-"
        print(f"{ctx_str:<12} {p_avg:<10} {p_max:<10} {d_avg:<10} {d_max:<10}")

    print()
    print("="*70)
    print("Benchmark 完成!")


if __name__ == "__main__":
    main()
