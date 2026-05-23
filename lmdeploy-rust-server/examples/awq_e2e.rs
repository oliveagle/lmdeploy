//! AWQ End-to-End Inference Example
//!
//! This example demonstrates loading an AWQ quantized model and running inference.
//!
//! Usage:
//!   cargo run --release --example awq_e2e -- --model /path/to/awq/model

use std::time::Instant;

use lmdeploy_server::model::GenerationParams;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ".to_string()
    };

    println!("=== AWQ End-to-End Inference Test ===");
    println!("Model path: {}", model_path);
    println!();

    // Check if model exists
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: Model path does not exist: {}", model_path);
        eprintln!("Please provide a valid model path:");
        eprintln!("  cargo run --release --example awq_e2e -- /path/to/awq/model");
        return Ok(());
    }

    // Load the model
    println!("1. Loading AWQ model...");
    let start = Instant::now();
    let engine = lmdeploy_server::model::cpp_engine::TurboMindCEngine::new(&model_path).await?;
    let load_time = start.elapsed();
    println!("   Model loaded in {:.2}s", load_time.as_secs_f64());

    // Check model info
    let info = engine.info();
    println!();
    println!("2. Model Info:");
    println!("   Name: {}", info.name);
    println!("   Engine: {}", info.engine_type.as_str());
    println!("   Quant Policy: {} (4=AWQ)", info.quant_policy);
    println!("   Hidden Size: {:?}", info.hidden_size);
    println!("   State: {:?}", info.state);

    // Test tokenizer
    println!();
    println!("3. Testing tokenizer...");
    if let Some(tokenizer) = engine.tokenizer() {
        let text = "Hello, world!";
        let encoded = tokenizer.encode(text, false, false)?;
        println!("   Text: '{}'", text);
        println!("   Tokens: {:?}", encoded);
        println!("   Vocab size: {}", tokenizer.vocab_size());
    }

    // Run inference
    println!();
    println!("4. Running inference...");
    let prompts = vec![
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        println!();
        println!("   Test {}:", i + 1);
        println!("   Prompt: '{}'", prompt);

        let start = Instant::now();
        let params = GenerationParams {
            max_tokens: Some(50),
            ..Default::default()
        };
        let (text, num_tokens, elapsed_ms) = engine.generate_with_metrics(prompt, params).await;
        let total_time = start.elapsed();

        println!("   Output: '{}'", text.trim());
        println!("   Tokens generated: {}", num_tokens);
        println!("   Inference time: {:.2}ms", elapsed_ms);
        println!("   Total time: {:.2}s", total_time.as_secs_f64());
        if num_tokens > 0 && elapsed_ms > 0.0 {
            println!("   Speed: {:.1} tokens/s", (num_tokens as f64) / (elapsed_ms / 1000.0));
        }
    }

    // Test embeddings
    println!();
    println!("5. Testing embeddings...");
    let embed_text = "Hello, world!";
    let start = Instant::now();
    let embedding = engine.embed(embed_text, Some(768)).await;
    let embed_time = start.elapsed();
    println!("   Text: '{}'", embed_text);
    println!("   Embedding dim: {}", embedding.len());
    println!("   First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
    println!("   Time: {:.2}ms", embed_time.as_secs_f64() * 1000.0);

    println!();
    println!("=== All tests completed successfully! ===");

    Ok(())
}
