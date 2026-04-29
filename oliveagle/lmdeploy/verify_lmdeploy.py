#!/usr/bin/env python3
"""
LMDeploy 功能验证 - 验证 prefill 速度和生成质量
"""
import time
import random
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

MODEL = '/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ'

# 真实测试文本（中英文混合）
TEST_TEXTS = [
    # 短文本 (~500 tokens)
    "人工智能是计算机科学的一个重要分支，致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。人工智能研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。当前的大语言模型通过海量数据训练，展现出了前所未有的语言理解和生成能力。",

    # 中文本 (~2000 tokens)
    """深度学习是机器学习的一个子领域，它源于人工神经网络的研究。深度学习模型通过多层非线性变换来学习数据的层次化表示，已经在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。

The development of deep learning has fundamentally transformed how we approach complex problems in computer science. Convolutional neural networks (CNNs) have revolutionized image recognition, while transformer architectures have become the dominant paradigm in natural language processing.

深度学习的核心是可微分编程思想：通过反向传播算法计算损失函数对模型参数的梯度，然后使用梯度下降类优化算法更新参数。常用的优化器包括 SGD、Adam、AdamW 等。正则化技术如 Dropout、BatchNorm、WeightDecay 等对于训练深度网络至关重要。""",

    # 长文本 (~8000 tokens) - 重复拼接
    None,
]

# 生成长文本
long_parts = []
for _ in range(20):
    part = random.choice([
        "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习模式和规律，而无需进行显式编程。监督学习、无监督学习和强化学习是机器学习的三大主要范式。深度学习作为机器学习的一个分支，通过多层神经网络实现了强大的特征提取和表示学习能力。",
        "Machine learning algorithms have been widely applied in various domains, from recommendation systems and fraud detection to medical diagnosis and autonomous driving. The success of machine learning depends on the availability of high-quality training data and appropriate feature engineering techniques.",
        "计算机视觉是让计算机看见和理解图像的技术，包括图像分类、目标检测、图像分割、人脸识别等任务。卷积神经网络通过局部感受野、权值共享和空间下采样等机制，大幅减少了模型参数数量，同时保持了出色的特征提取能力。",
        "The transformer architecture, introduced in the seminal paper Attention Is All You Need, has become the foundation of modern large language models. Self-attention mechanisms allow the model to process sequences in parallel and capture long-range dependencies more effectively than recurrent neural networks.",
    ])
    long_parts.append(part)
TEST_TEXTS[2] = "\n\n".join(long_parts)


def count_tokens(text, tokenizer):
    """使用 tokenizer 计算真实 token 数"""
    return len(tokenizer.encode(text))


def test_inference(pipe, tokenizer, name: str, text: str, max_new_tokens: int = 128):
    """测试单次推理"""
    print(f"\n{'='*60}")
    print(f"测试：{name}")
    print(f"{'='*60}")

    # 计算真实 token 数
    input_tokens = count_tokens(text, tokenizer)
    print(f"输入文本字符数：{len(text):,}")
    print(f"输入 token 数：{input_tokens:,}")

    # Prefill + Decode 计时
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)

    print(f"\n开始推理 (max_new_tokens={max_new_tokens})...")
    start = time.perf_counter()
    response = pipe.infer(text, gen_config=gen_config)
    elapsed = time.perf_counter() - start

    # 统计结果
    output_tokens = len(response.token_ids) if hasattr(response, 'token_ids') else 0
    total_time = elapsed

    # 估算 prefill 和 decode 时间
    # 假设 decode 每 token 约 23ms (43 tok/s)
    estimated_decode_time = output_tokens * 0.023
    estimated_prefill_time = max(0.001, elapsed - estimated_decode_time)

    prefill_speed = input_tokens / estimated_prefill_time if estimated_prefill_time > 0 else 0
    decode_speed = output_tokens / estimated_decode_time if estimated_decode_time > 0 else 0

    print(f"\n结果统计:")
    print(f"  总耗时：{total_time:.3f}s")
    print(f"  输出 token 数：{output_tokens:,}")
    print(f"  估算 Prefill 时间：{estimated_prefill_time:.4f}s -> {prefill_speed:.0f} tok/s")
    print(f"  估算 Decode 时间：{estimated_decode_time:.4f}s -> {decode_speed:.1f} tok/s")

    # 检查生成质量
    generated_text = response.text if hasattr(response, 'text') else str(response)
    print(f"\n生成内容预览 (前 300 字符):")
    print(f"  {generated_text[:300].replace(chr(10), ' ')}...")

    # 检查是否有重复、乱码等异常
    has_repeat = len(generated_text) > 100 and generated_text[:100] in generated_text[100:]
    has_gibberish = '' in generated_text

    print(f"\n质量检查:")
    print(f"  重复检测：{'有重复' if has_repeat else '正常'}")
    print(f"  乱码检测：{'有乱码' if has_gibberish else '正常'}")

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_time': total_time,
        'prefill_speed': prefill_speed,
        'decode_speed': decode_speed,
        'has_repeat': has_repeat,
        'has_gibberish': has_gibberish,
    }


def main():
    print("="*60)
    print("LMDeploy 功能验证测试")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    backend_config = TurbomindEngineConfig(
        session_len=16384,  # 16K context 测试
        max_batch_size=1,
        cache_max_entry_count=0.8,
        kv_cache_dtype='int4',
    )

    pipe = pipeline(
        model_path=MODEL,
        backend_config=backend_config,
        tp=1,
    )

    # 获取 tokenizer
    from lmdeploy import Tokenizer
    tokenizer = Tokenizer(MODEL)

    results = []

    # 测试 1: 短文本
    r1 = test_inference(pipe, tokenizer, "短文本 (~500 tokens)", TEST_TEXTS[0], max_new_tokens=64)
    results.append(('短文本', r1))

    # 测试 2: 中文本
    r2 = test_inference(pipe, tokenizer, "中文本 (~2000 tokens)", TEST_TEXTS[1], max_new_tokens=128)
    results.append(('中文本', r2))

    # 测试 3: 长文本
    r3 = test_inference(pipe, tokenizer, "长文本 (~8000 tokens)", TEST_TEXTS[2], max_new_tokens=256)
    results.append(('长文本', r3))

    # 测试 4: 问答测试
    print(f"\n{'='*60}")
    print("测试：问答能力")
    print(f"{'='*60}")
    qa_prompt = "请用 200 字左右解释什么是量子纠缠。"
    r4 = test_inference(pipe, tokenizer, "问答测试", qa_prompt, max_new_tokens=256)
    results.append(('问答', r4))

    # 测试 5: 代码生成
    print(f"\n{'='*60}")
    print("测试：代码生成")
    print(f"{'='*60}")
    code_prompt = "请用 Python 写一个快速排序函数，并添加简要注释。"
    r5 = test_inference(pipe, tokenizer, "代码生成", code_prompt, max_new_tokens=300)
    results.append(('代码', r5))

    # 汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    print(f"{'测试':<10} {'输入 tokens':>12} {'输出 tokens':>12} {'Prefill':>12} {'Decode':>10} {'质量':>8}")
    print("-"*60)

    for name, r in results:
        quality = "OK" if not r['has_repeat'] and not r['has_gibberish'] else "FAIL"
        print(f"{name:<10} {r['input_tokens']:>12,} {r['output_tokens']:>12,} {r['prefill_speed']:>10,.0f} {r['decode_speed']:>10,.1f} {quality:>8}")

    print("\n所有测试完成！")

    pipe.close()


if __name__ == "__main__":
    main()
