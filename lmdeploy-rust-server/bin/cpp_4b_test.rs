//! Quick test for C++ engine with smaller 4B model
use lmdeploy_server::model::cpp_engine::TurboMindCEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_path = "/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3-4B";

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

    let gen_start = std::time::Instant::now();
    let result = engine.generate(prompt, 32).await;
    let gen_time = gen_start.elapsed();

    println!("Output: {}", result);
    println!("Generation time: {:.2}s", gen_time.as_secs_f64());

    println!("SUCCESS: Pure C++ engine E2E inference working!");

    Ok(())
}
