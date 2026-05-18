//! LMDeploy Performance Benchmark Tool
//!
//! Runs comprehensive performance benchmarks on LMDeploy TurboMind engine:
//! - Multiple context lengths (1K, 4K, 8K tokens)
//! - TTFT (Time To First Token) measurement
//! - Prefill and decode speed tracking
//! - GPU memory usage monitoring
//!
//! Usage:
//!   cargo run --example benchmark -- <model_path>
//!
//! Example:
//!   cargo run --example benchmark -- /mnt/eaget-4tb/modelscope_models/tclf00/Qwen3___6-35B-A3B-AWQ

use std::sync::Arc;
use std::time::Instant;

use lmdeploy_server::model::benchmark::{BenchmarkConfig, BenchmarkRunner};
use lmdeploy_server::model::TurboMindEngine;

/// Benchmark scenarios with different context lengths
const OUTPUT_LENGTH: usize = 512;
const ITERATIONS: usize = 3;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Parse model path from CLI arguments
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ".to_string());

    println!("=== LMDeploy Performance Benchmark ===");
    println!("Model: {}", model_path);
    println!("Context lengths: 1K, 4K, 8K tokens");
    println!("Output length: {} tokens", OUTPUT_LENGTH);
    println!("Iterations: {}", ITERATIONS);
    println!("======================================\n");

    // Initialize engine
    println!("Initializing TurboMind engine...");
    let engine_start = Instant::now();
    let engine = Arc::new(TurboMindEngine::new(&model_path).await?);
    let engine_init_time = engine_start.elapsed();
    println!("Engine initialized in {:.2}s\n", engine_init_time.as_secs_f64());

    // Get initial GPU memory
    let initial_memory = get_gpu_memory().unwrap_or(0.0);

    // Create benchmark config
    let config = BenchmarkConfig {
        context_lengths: vec![1024, 4096, 8192],
        output_length: OUTPUT_LENGTH,
        iterations: ITERATIONS,
        warmup_iterations: 1,
    };

    // Run benchmarks
    let runner = BenchmarkRunner::new(engine, config);
    let report = runner.run().await?;

    // Get final GPU memory
    let final_memory = get_gpu_memory().unwrap_or(0.0);
    let memory_used = final_memory - initial_memory;

    // Print summary
    println!("\n=== Benchmark Summary ===");
    println!("GPU Memory Used: {:.2} GB", memory_used);
    println!();

    for summary in &report.summaries {
        let context_label = if summary.context_length >= 1024 {
            format!("{}K", summary.context_length / 1024)
        } else {
            format!("{}", summary.context_length)
        };

        println!("{} context:", context_label);
        println!("  Avg TTFT:        {:.2} ms", summary.avg_ttft_ms);
        println!("  Avg Prefill:     {:.2} tokens/s", summary.avg_prefill_speed_tps);
        println!("  Avg Decode:      {:.2} tokens/s", summary.avg_decode_speed_tps);
        println!("  Avg Total Time:  {:.2} ms", summary.avg_total_time_ms);
    }

    // Save results to JSON
    save_results(&report, memory_used)?;

    Ok(())
}

/// Get GPU memory usage in GB using nvidia-smi
fn get_gpu_memory() -> Option<f64> {
    use std::process::Command;

    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if output.status.success() {
        let memory_mb = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<f64>()
            .unwrap_or(0.0);
        Some(memory_mb / 1024.0)
    } else {
        None
    }
}

/// Save results to JSON file
fn save_results(report: &lmdeploy_server::model::benchmark::BenchmarkReport, memory_used_gb: f64) -> Result<(), Box<dyn std::error::Error>> {
    let output = serde_json::json!({
        "model": report.model_path,
        "engine": report.engine_name,
        "gpu_memory_used_gb": memory_used_gb,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "summary": report.summaries,
        "all_results": report.results,
    });

    let filename = format!("benchmark_results_{}.json",
        chrono::Utc::now().format("%Y%m%d_%H%M%S"));

    std::fs::write(&filename, serde_json::to_string_pretty(&output)?)?;
    println!("\nResults saved to: {}", filename);

    Ok(())
}
