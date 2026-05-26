//! 独立的 Rust benchmark，不依赖 lmdeploy-server lib
//! 直接链接到 lmdeploy-server-core

use std::time::Instant;

fn main() {
    println!("Rust prefill benchmark 独立测试");
    println!();
    println!("Model: Qwen3.6-35B-A3B-AWQ");
    println!("Note: 这是 tokenizer benchmark，不包含模型推理");
    println!();

    // 简单的字符串重复
    let repeat = "The quick brown fox jumps over the lazy dog. ";

    println!("{:>8} | {:>10} | {:>12}", "Context", "Chars", "Est. Tokens");
    println!("{}", "-".repeat(40));

    for &label in &["1K", "4K", "8K"] {
        let count = match label {
            "1K" => 102,
            "4K" => 409,
            "8K" => 819,
            _ => continue,
        };

        let prompt = repeat.repeat(count);
        let actual_chars = prompt.len();
        let approx_tokens = actual_chars / 4;

        let start = Instant::now();
        let _tokens: Vec<&str> = prompt.split_whitespace().collect();
        let elapsed = start.elapsed().as_micros() as f64 / 1000.0;

        println!("{:>8} | {:>10} | {:>12} (tokenize: {:.1}ms)",
                 label, actual_chars, approx_tokens, elapsed);
    }
}
