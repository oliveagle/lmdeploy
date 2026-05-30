use std::time::Instant;
use lmdeploy_server::model::cpp_engine::{GenerationParams, TurboMindCEngine};

const MODEL_PATH: &str = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ";
const NUM_REQUESTS: usize = 5;
const WARMUP: usize = 2;

#[tokio::test]
async fn benchmark_rust_prefill() {
    println!("Rust Prefill Benchmark");
    println!("Model: {}", MODEL_PATH);
    println!();

    println!("Loading model...");
    let engine = TurboMindCEngine::new(MODEL_PATH).await.expect("Failed to load model");
    println!("Model loaded successfully\n");

    let scenarios = vec![
        ("512", 512),
        ("1024", 1024),
        ("2048", 2048),
        ("4096", 4096),
        ("8192", 8192),
    ];

    println!("{:=<80}", "");
    println!("{:^80}", "PREFILL BENCHMARK RESULTS");
    println!("{:=<80}", "");

    for (name, target_tokens) in &scenarios {
        // Generate prompt of target length (repeated tokens)
        let base_prompt = "hello world test data ";
        let repeat_count = *target_tokens * 4;
        let prompt = base_prompt.repeat(repeat_count / base_prompt.len() + 1);

        println!("\n--- {} (target {} tokens) ---", name, target_tokens);

        // Warmup
        for _ in 0..WARMUP {
            let _ = engine.generate("hi", GenerationParams::default()).await;
        }

        // Benchmark
        let mut ttfts = Vec::with_capacity(NUM_REQUESTS);
        for i in 0..NUM_REQUESTS {
            let start = Instant::now();
            let _ = engine.generate(
                &prompt,
                GenerationParams {
                    max_tokens: Some(1),
                    ..Default::default()
                },
            ).await;
            let ttft = start.elapsed().as_secs_f64() * 1000.0;
            ttfts.push(ttft);
            println!("  Run {}: TTFT={:.1}ms", i + 1, ttft);
        }

        if !ttfts.is_empty() {
            let avg = ttfts.iter().sum::<f64>() / ttfts.len() as f64;
            let min = ttfts.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = ttfts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Estimate token count (rough approximation: ~4 chars per token for English)
            let estimated_tokens = prompt.len() / 4;
            let prefill_tps_avg = estimated_tokens as f64 / avg * 1000.0;
            let prefill_tps_max = estimated_tokens as f64 / min * 1000.0;

            println!("  TTFT: avg={:.1}ms, min={:.1}ms, max={:.1}ms", avg, min, max);
            println!("  Prefill: avg={:.0} tok/s, max={:.0} tok/s (est. {} input tokens)", prefill_tps_avg, prefill_tps_max, estimated_tokens);
        }
    }

    println!("\n{:=<80}", "");
    println!("Benchmark complete");
}
