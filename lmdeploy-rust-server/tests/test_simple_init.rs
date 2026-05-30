//! Test for simple model initialization to identify crash
use lmdeploy_server::model::cpp_engine::TurboMindCEngine;
use std::sync::Arc;

#[tokio::test]
async fn test_simple_init() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let model_path = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ";

    println!("Starting model initialization...");
    match TurboMindCEngine::new(model_path).await {
        Ok(_engine) => println!("SUCCESS: Model initialized!"),
        Err(e) => println!("ERROR: Model initialization failed: {}", e),
    }
}
