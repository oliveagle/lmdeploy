//! End-to-end test for pure Rust + C++ inference
//!
//! This example verifies the complete inference path:
//! 1. Model loading from HuggingFace safetensors
//! 2. Tokenizer initialization
//! 3. C++ engine initialization
//! 4. Inference execution
//! 5. Output decoding
//!
//! Usage:
//!   cargo run --example e2e_test -- [model_path]
//!
//! Example:
//!   cargo run --example e2e_test -- /mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ

use std::time::Instant;

use lmdeploy_server::model::cpp_engine::{EngineType, TurboMindCEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            "/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ".to_string()
        });

    println!("=== LMDeploy Pure C++ E2E Test ===");
    println!("Model: {}", model_path);

    // Phase 1: Model loading
    println!("\n[Phase 1] Loading C++ engine...");
    let load_start = Instant::now();
    let engine = match TurboMindCEngine::new(&model_path).await {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAIL: Engine initialization failed: {:?}", e);
            return Err(format!("Engine init failed: {:?}", e).into());
        }
    };
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("OK: Engine loaded in {:.2}ms", load_ms);
    println!("  Engine type: {:?}", engine.engine_type.as_str());
    println!("  Model name: {}", engine.info().name);
    println!("  AWQ quantization: {}", engine.info().quant_policy);

    // Phase 2: Tokenizer
    println!("\n[Phase 2] Verifying tokenizer...");
    let tokenizer = engine.tokenizer().ok_or("Tokenizer not available")?;
    println!("OK: Tokenizer loaded (vocab size: {})", tokenizer.vocab_size());

    // Phase 3: Simple inference
    println!("\n[Phase 3] Running inference...");
    let prompt = "The capital of France is";
    println!("  Prompt: \"{}\"", prompt);

    let inference_start = Instant::now();
    let (output, num_tokens, _elapsed_ms) = engine.generate_with_metrics(prompt, 32).await;
    let total_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

    if output.is_empty() {
        eprintln!("FAIL: Empty output");
        return Err("Inference returned empty output".into());
    }

    println!("OK: Inference completed");
    println!("  Output: \"{}\"", output.trim());
    println!("  Tokens: {}", num_tokens);
    println!("  Time: {:.2}ms", total_ms);
    if num_tokens > 0 {
        println!("  Speed: {:.2} tokens/s", (num_tokens as f64) / (total_ms / 1000.0));
    }

    // Phase 4: Longer generation
    println!("\n[Phase 4] Longer generation (128 tokens)...");
    let long_prompt = "Write a short Python function to calculate factorial:";
    println!("  Prompt: \"{}\"", long_prompt);

    let start = Instant::now();
    let (output2, tokens2, _elapsed2) = engine.generate_with_metrics(long_prompt, 128).await;
    let total2 = start.elapsed().as_secs_f64() * 1000.0;

    if output2.is_empty() {
        eprintln!("FAIL: Empty output for longer generation");
        return Err("Longer generation returned empty output".into());
    }

    println!("OK: Longer generation completed");
    println!("  Output preview: \"{}\"...",
        output2.chars().take(100).collect::<String>());
    println!("  Total tokens: {}", tokens2);
    println!("  Time: {:.2}ms", total2);
    if tokens2 > 0 {
        println!("  Speed: {:.2} tokens/s", (tokens2 as f64) / (total2 / 1000.0));
    }

    // Summary
    println!("\n=== E2E Test Summary ===");
    println!("Engine type: {}", EngineType::PureCpp.as_str());
    println!("Model: {}", model_path);
    println!("Load time: {:.2}ms", load_ms);
    println!("Short gen: {:.2}ms ({} tokens)", total_ms, num_tokens);
    println!("Long gen: {:.2}ms ({} tokens)", total2, tokens2);
    println!("Result: PASS");
    println!("========================");

    Ok(())
}
