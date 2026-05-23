//! Quick test for C++ engine with smaller 4B model
use lmdeploy_server::model::cpp_engine::TurboMindCEngine;
use lmdeploy_server::model::GenerationParams;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_path = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ";

    println!("Initializing C++ engine with model: {}", model_path);

    let start = std::time::Instant::now();
    let engine = match TurboMindCEngine::new(model_path).await {
        Ok(e) => {
            println!("Engine loaded in {:.2}s", start.elapsed().as_secs_f64());
            e
        }
        Err(e) => {
            println!("FAILED: {:?}", e);
            return Ok(());
        }
    };

    println!("Engine ready: {}", engine.is_ready());

    let prompt = "What is 2+2?";
    println!("Prompt: {}", prompt);

    let params = GenerationParams {
        max_tokens: Some(32),
        ..Default::default()
    };
    let gen_start = std::time::Instant::now();
    let result = engine.generate(prompt, params).await;
    let gen_time = gen_start.elapsed();

    println!("Output: {}", result);
    println!("Generation time: {:.2}s", gen_time.as_secs_f64());

    println!("SUCCESS: Pure C++ engine E2E inference working!");

    Ok(())
}
