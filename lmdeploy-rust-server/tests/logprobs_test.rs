//! Test logprobs functionality
//!
//! This test verifies that logprobs are correctly extracted from the C++ engine
//! and returned through the API.

use lmdeploy_server::model::cpp_engine::TurboMindCEngine;
use lmdeploy_server::model::GenerationParams;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".to_string());

    println!("Testing logprobs with model: {}", model_path);

    let engine = match TurboMindCEngine::new(&model_path).await {
        Ok(e) => e,
        Err(e) => {
            println!("FAILED to load engine: {:?}", e);
            return Ok(());
        }
    };

    println!("Engine ready: {}", engine.is_ready());

    let prompt = "What is the capital of France?";
    println!("Prompt: {}", prompt);

    // Test 1: Generate with logprobs=true
    println!("\n=== Test 1: logprobs=true ===");
    let params = GenerationParams {
        max_tokens: Some(32),
        logprobs: Some(true),
        top_logprobs: Some(5),
        ..Default::default()
    };

    let (text, _num_tokens, _elapsed_ms, logprobs) =
        engine.generate_with_logprobs(prompt, params).await;

    println!("Output: {}", text);
    if let Some(lp) = logprobs {
        println!("Logprobs received: {} entries", lp.len());
        for (i, entry) in lp.iter().take(3).enumerate() {
            println!(
                "  Token {}: '{}' (logprob={:.4}), top_logprobs: {}",
                i,
                entry.token,
                entry.logprob,
                entry.top_logprobs.len()
            );
            for (j, top) in entry.top_logprobs.iter().take(3).enumerate() {
                println!("    - Top {}: '{}' ({:.4})", j, top.token, top.logprob);
            }
        }
        println!("Test 1: PASSED");
    } else {
        println!("Test 1: FAILED - No logprobs returned");
    }

    // Test 2: Generate without logprobs
    println!("\n=== Test 2: logprobs=false ===");
    let params = GenerationParams {
        max_tokens: Some(16),
        logprobs: Some(false),
        ..Default::default()
    };

    let (text2, _, _, logprobs2) = engine.generate_with_logprobs(prompt, params).await;

    println!("Output: {}", text2);
    if logprobs2.is_none() {
        println!("Test 2: PASSED - No logprobs returned when logprobs=false");
    } else {
        println!("Test 2: FAILED - Unexpected logprobs returned");
    }

    // Test 3: Different top_logprobs values
    println!("\n=== Test 3: top_logprobs=10 ===");
    let params = GenerationParams {
        max_tokens: Some(8),
        logprobs: Some(true),
        top_logprobs: Some(10),
        ..Default::default()
    };

    let (text3, _, _, logprobs3) = engine.generate_with_logprobs("Hello", params).await;

    println!("Output: {}", text3);
    if let Some(lp) = logprobs3 {
        let min_top = lp.iter().map(|e| e.top_logprobs.len()).min().unwrap_or(0);
        let max_top = lp.iter().map(|e| e.top_logprobs.len()).max().unwrap_or(0);
        println!("Top logprobs range: {} to {}", min_top, max_top);
        if max_top <= 10 {
            println!("Test 3: PASSED");
        } else {
            println!("Test 3: FAILED - Expected max 10 top_logprobs, got {}", max_top);
        }
    } else {
        println!("Test 3: FAILED - No logprobs returned");
    }

    println!("\n=== All tests completed ===");

    Ok(())
}
