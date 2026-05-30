//! Minimal test for model initialization
use lmdeploy_server::model::cpp_engine::TurboMindCEngine;
use std::sync::Arc;
use std::env;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".into());

    println!("Testing model loading: {}", model_path);

    match TurboMindCEngine::new(&model_path).await {
        Ok(engine) => {
            println!("SUCCESS: Model loaded!");

            // Test a simple inference
            let params = lmdeploy_server::model::cpp_engine::GenerationParams {
                max_tokens: Some(10),
                temperature: Some(0.0),
                ..Default::default()
            };

            println!("Testing inference...");
            let result = engine.generate("Hello", params).await;
            println!("Inference result: {:?}", result);
        },
        Err(e) => {
            eprintln!("ERROR: Failed to load model: {}", e);
        }
    }
}