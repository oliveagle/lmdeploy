//! Minimal test to reproduce 2048 token stream issue

use std::sync::Arc;
use clap::Parser;
use futures::StreamExt;

use lmdeploy_server::model::cpp_engine::{GenerationParams, TurboMindCEngine};

/// Test 2048 token streaming
#[derive(Parser, Debug)]
#[command(name = "test_2048")]
struct Args {
    /// Model path
    #[arg(long)]
    model: Option<String>,
}

/// Generate a prompt with approximately target_token_count tokens
fn gen_prompt(target_token_count: usize) -> String {
    let repeat_text = "The quick brown fox jumps over the lazy dog. ";
    let chars_per_token = 4;
    let target_chars = target_token_count * chars_per_token;
    let repeats = (target_chars / repeat_text.len()) + 1;
    repeat_text.repeat(repeats)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let model_path = args.model.unwrap_or_else(|| {
        "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".to_string()
    });

    println!("Testing 2048 token streaming...");
    println!("Model: {}", model_path);

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // Load engine
    println!("Loading TurboMind engine...");
    let engine = Arc::new(TurboMindCEngine::new(&model_path).await?);
    println!("Engine loaded successfully\n");

    // Check tokenizer
    let prompt_2048 = gen_prompt(2048);
    let tokenizer = engine.tokenizer().ok_or("Tokenizer not available")?;
    let token_ids = tokenizer.encode(&prompt_2048, false, false)?;
    println!("Generated 2048 token prompt:");
    println!("  Length: {} chars", prompt_2048.len());
    println!("  Tokens: {} actual tokens\n", token_ids.len());

    // Test streaming
    println!("Testing stream...");
    let params = GenerationParams {
        max_tokens: Some(10),  // Just get a few tokens
        temperature: Some(0.0),
        top_p: Some(1.0),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let mut stream = engine.generate_stream(&prompt_2048, params).await;
    let mut token_count = 0;
    let mut first_token_time = None;

    while let Some((token_id, token_text)) = stream.next().await {
        let elapsed = start.elapsed();
        if token_count == 0 {
            first_token_time = Some(elapsed);
            println!("  First token: {:?} ({} ms)", token_text, elapsed.as_millis());
        } else {
            println!("  Token {}: {:?}", token_count, token_text);
        }
        token_count += 1;
        if token_count >= 5 {
            break;
        }
    }

    if token_count == 0 {
        println!("ERROR: NO TOKENS RECEIVED!");
        println!("\nNow testing non-streaming generate...");
        let params_nostream = GenerationParams {
            max_tokens: Some(10),
            temperature: Some(0.0),
            top_p: Some(1.0),
            ..Default::default()
        };
        let text = engine.generate(&prompt_2048, params_nostream).await;
        println!("Non-streaming result: {}", text);
    } else {
        println!("OK: Received {} tokens", token_count);
    }

    Ok(())
}
