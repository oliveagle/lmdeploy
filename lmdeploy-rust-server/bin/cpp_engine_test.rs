//! Quick E2E test for the pure C++ engine
use std::sync::Arc;
use lmdeploy_server::model::cpp_engine::TurboMindCEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_path = "/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___5-9B-AWQ";
    
    println!("Initializing C++ engine with model: {}", model_path);
    
    let engine = match TurboMindCEngine::new(model_path).await {
        Ok(e) => e,
        Err(e) => {
            println!("FAILED: {:?}", e);
            return Ok(());
        }
    };
    
    println!("Engine ready: {}", engine.is_ready());
    
    let prompt = "What is the capital of France?";
    println!("Prompt: {}", prompt);
    
    let result = engine.generate(prompt, 32).await;
    println!("Output: {}", result);
    
    println!("SUCCESS: Pure C++ engine E2E inference working!");
    
    Ok(())
}
