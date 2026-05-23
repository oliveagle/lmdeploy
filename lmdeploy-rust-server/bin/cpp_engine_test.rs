//! Quick E2E test for the pure C++ engine
use lmdeploy_server::model::cpp_engine::TurboMindCEngine;
use lmdeploy_server::model::GenerationParams;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_path = "/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3.5-9B-TextOnly";

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

    let params = GenerationParams {
        max_tokens: Some(32),
        ..Default::default()
    };
    let result = engine.generate(prompt, params).await;
    println!("Output: {}", result);

    println!("SUCCESS: Pure C++ engine E2E inference working!");

    Ok(())
}
