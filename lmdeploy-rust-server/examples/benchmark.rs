//! LMDeploy Performance Benchmark Tool
//!
//! Runs comprehensive performance benchmarks on LMDeploy TurboMind engine:
//! - Multiple context lengths (1K, 4K, 8K, 16K, 32K tokens)
//! - TTFT (Time To First Token) measurement
//! - Prefill and decode speed tracking
//! - GPU memory usage monitoring
//! - Support for AWQ quantized models
//! - Concurrent request testing
//! - Multiple output formats (JSON, table)
//!
//! Usage:
//!   cargo run --example benchmark -- [OPTIONS] <model_path>
//!
//! Examples:
//!   # AWQ model with default settings
//!   cargo run --example benchmark -- /path/to/awq-model
//!
//!   # Non-AWQ model with custom context lengths
//!   cargo run --example benchmark -- --context-lengths 1024 4096 8192 /path/to/model
//!
//!   # Concurrent request testing
//!   cargo run --example benchmark -- --concurrent 4 --output-length 128 /path/to/model
//!
//!   # Table output format
//!   cargo run --example benchmark -- --format table /path/to/model

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use lmdeploy_server::model::benchmark::{BenchmarkConfig, BenchmarkRunner};
use lmdeploy_server::model::GenerationParams;
use lmdeploy_server::model::TurboMindCEngine;
use tabled::{Table, Tabled};

/// Benchmark CLI arguments
#[derive(Parser, Debug)]
#[command(name = "lmdeploy-benchmark")]
#[command(about = "LMDeploy TurboMind performance benchmark tool", long_about = None)]
struct Args {
    /// Model path (HuggingFace format)
    #[arg(value_name = "MODEL_PATH")]
    model_path: String,

    /// Context lengths to test (in tokens)
    #[arg(long = "context-lengths", value_name = "TOKENS", value_delimiter = ' ', default_values_t = vec![1024, 4096, 8192, 16384, 32768])]
    context_lengths: Vec<usize>,

    /// Output length (in tokens)
    #[arg(long = "output-length", default_value_t = 512)]
    output_length: usize,

    /// Number of iterations per test
    #[arg(long = "iterations", default_value_t = 3)]
    iterations: usize,

    /// Warmup iterations (not counted in results)
    #[arg(long = "warmup", default_value_t = 1)]
    warmup_iterations: usize,

    /// Number of concurrent requests for concurrency test
    #[arg(long = "concurrent", default_value_t = 1)]
    concurrent_requests: usize,

    /// Quantization type (auto-detect, awq, fp8, w8a8, none)
    #[arg(long = "quantization", default_value = "auto")]
    quantization: String,

    /// Output format (json, table, both)
    #[arg(long = "format", default_value = "both")]
    output_format: String,

    /// Output file path (for JSON format)
    #[arg(long = "output", value_name = "FILE")]
    output_file: Option<String>,

    /// Verbose output
    #[arg(long, short = 'v')]
    verbose: bool,
}

/// Benchmark result entry for table output
#[derive(Tabled)]
struct BenchmarkTableRow {
    #[tabled(rename = "Context")]
    context_label: String,
    #[tabled(rename = "TTFT (ms)")]
    ttft_ms: String,
    #[tabled(rename = "Prefill (t/s)")]
    prefill_tps: String,
    #[tabled(rename = "Decode (t/s)")]
    decode_tps: String,
    #[tabled(rename = "Total (ms)")]
    total_ms: String,
}

/// Concurrent test result
#[derive(Debug, Clone, serde::Serialize)]
struct ConcurrentResult {
    num_requests: usize,
    total_time_ms: f64,
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    throughput_rps: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(log_level).init();

    println!("=== LMDeploy Performance Benchmark ===");
    println!("Model: {}", args.model_path);
    println!("Quantization: {}", args.quantization);
    println!("Context lengths: {}", format_context_lengths(&args.context_lengths));
    println!("Output length: {} tokens", args.output_length);
    println!("Iterations: {}", args.iterations);
    println!("Concurrent requests: {}", args.concurrent_requests);
    println!("======================================\n");

    // Detect quantization type
    let quant_type = detect_quantization(&args.model_path, &args.quantization)?;
    println!("Detected quantization: {}\n", quant_type);

    // Initialize engine
    println!("Initializing TurboMind engine...");
    let engine_start = Instant::now();
    let engine = Arc::new(TurboMindCEngine::new(&args.model_path).await?);
    let engine_init_time = engine_start.elapsed();
    println!("Engine initialized in {:.2}s\n", engine_init_time.as_secs_f64());

    // Get initial GPU memory
    let initial_memory = get_gpu_memory().unwrap_or(0.0);

    // Run sequential benchmarks
    let report = run_sequential_benchmarks(&engine, &args).await?;

    // Get final GPU memory
    let final_memory = get_gpu_memory().unwrap_or(0.0);
    let memory_used = final_memory - initial_memory;

    // Run concurrent benchmark if requested
    let concurrent_result = if args.concurrent_requests > 1 {
        println!("\n=== Concurrent Request Benchmark ===");
        let result = run_concurrent_benchmark(&engine, args.concurrent_requests, args.output_length).await?;
        println!("Concurrent requests: {}", result.num_requests);
        println!("Total time: {:.2} ms", result.total_time_ms);
        println!("Average latency: {:.2} ms", result.avg_time_ms);
        println!("Min latency: {:.2} ms", result.min_time_ms);
        println!("Max latency: {:.2} ms", result.max_time_ms);
        println!("Throughput: {:.2} req/s", result.throughput_rps);
        Some(result)
    } else {
        None
    };

    // Output results
    output_results(&report, memory_used, &concurrent_result, &args)?;

    Ok(())
}

/// Detect quantization type from model path or config
fn detect_quantization(model_path: &str, quant_arg: &str) -> Result<String, Box<dyn std::error::Error>> {
    if quant_arg != "auto" {
        return Ok(quant_arg.to_string());
    }

    // Check for AWQ model
    let config_path = std::path::Path::new(model_path).join("config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(quant_method) = config.get("quant_method").and_then(|v| v.as_str()) {
                    return Ok(format!("AWQ ({})", quant_method));
                }
            }
        }
    }

    // Check path hints
    let path_lower = model_path.to_lowercase();
    if path_lower.contains("awq") || path_lower.contains("4bit") {
        return Ok("AWQ (detected from path)".to_string());
    }
    if path_lower.contains("fp8") {
        return Ok("FP8".to_string());
    }
    if path_lower.contains("w8a8") {
        return Ok("W8A8".to_string());
    }

    Ok("None (FP16/BF16)".to_string())
}

/// Format context lengths for display
fn format_context_lengths(lengths: &[usize]) -> String {
    lengths.iter()
        .map(|&l| if l >= 1024 { format!("{}K", l / 1024) } else { format!("{}", l) })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Run sequential benchmarks
async fn run_sequential_benchmarks(
    engine: &Arc<TurboMindCEngine>,
    args: &Args,
) -> Result<lmdeploy_server::model::benchmark::BenchmarkReport, Box<dyn std::error::Error>> {
    println!("=== Running Sequential Benchmarks ===");

    let config = BenchmarkConfig {
        context_lengths: args.context_lengths.clone(),
        output_length: args.output_length,
        iterations: args.iterations,
        warmup_iterations: args.warmup_iterations,
    };

    let runner = BenchmarkRunner::new(Arc::clone(engine), config);
    let report = runner.run().await.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    Ok(report)
}

/// Run concurrent benchmark
async fn run_concurrent_benchmark(
    engine: &Arc<TurboMindCEngine>,
    concurrent_requests: usize,
    output_length: usize,
) -> Result<ConcurrentResult, Box<dyn std::error::Error>> {
    let prompt = "The quick brown fox jumps over the lazy dog. Please explain what this sentence means and why it is used in computing.";

    let mut handles = Vec::new();
    let start = Instant::now();

    for _ in 0..concurrent_requests {
        let engine_clone = Arc::clone(engine);
        let prompt = prompt.to_string();
        let handle = tokio::spawn(async move {
            let params = GenerationParams {
                max_tokens: Some(output_length),
                ..Default::default()
            };
            let req_start = Instant::now();
            let _result = engine_clone.generate(&prompt, params).await;
            Ok::<_, String>(req_start.elapsed().as_secs_f64() * 1000.0)
        });
        handles.push(handle);
    }

    let mut times = Vec::new();
    for handle in handles {
        let time = handle.await.map_err(|e| format!("Join error: {}", e))??;
        times.push(time);
    }

    let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let avg_time_ms = times.iter().sum::<f64>() / times.len() as f64;
    let min_time_ms = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time_ms = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let throughput_rps = (concurrent_requests as f64 * 1000.0) / total_time_ms;

    Ok(ConcurrentResult {
        num_requests: concurrent_requests,
        total_time_ms,
        avg_time_ms,
        min_time_ms,
        max_time_ms,
        throughput_rps,
    })
}

/// Output results in requested format
fn output_results(
    report: &lmdeploy_server::model::benchmark::BenchmarkReport,
    memory_used_gb: f64,
    concurrent_result: &Option<ConcurrentResult>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let format = args.output_format.to_lowercase();

    if format == "json" || format == "both" {
        output_json(report, memory_used_gb, concurrent_result, args)?;
    }

    if format == "table" || format == "both" {
        output_table(report, memory_used_gb, concurrent_result);
    }

    Ok(())
}

/// Output results as JSON
fn output_json(
    report: &lmdeploy_server::model::benchmark::BenchmarkReport,
    memory_used_gb: f64,
    concurrent_result: &Option<ConcurrentResult>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = serde_json::json!({
        "model": report.model_path,
        "engine": report.engine_name,
        "quantization": args.quantization,
        "gpu_memory_used_gb": memory_used_gb,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "config": {
            "context_lengths": args.context_lengths,
            "output_length": args.output_length,
            "iterations": args.iterations,
            "concurrent_requests": args.concurrent_requests,
        },
        "sequential": {
            "summary": report.summaries,
            "all_results": report.results,
        },
        "concurrent": concurrent_result,
    });

    let json_str = serde_json::to_string_pretty(&output)?;

    if let Some(ref file) = args.output_file {
        std::fs::write(file, json_str)?;
        println!("\nJSON results saved to: {}", file);
    } else if args.output_format == "json" {
        let filename = format!("benchmark_results_{}.json",
            chrono::Utc::now().format("%Y%m%d_%H%M%S"));
        std::fs::write(&filename, json_str)?;
        println!("\nJSON results saved to: {}", filename);
    } else {
        println!("\n=== JSON Output ===");
        println!("{}", json_str);
    }

    Ok(())
}

/// Output results as table
fn output_table(
    report: &lmdeploy_server::model::benchmark::BenchmarkReport,
    memory_used_gb: f64,
    concurrent_result: &Option<ConcurrentResult>,
) {
    println!("\n=== Benchmark Summary ===");
    println!("GPU Memory Used: {:.2} GB", memory_used_gb);
    println!();

    // Sequential results table
    let mut rows = Vec::new();
    for summary in &report.summaries {
        let context_label = if summary.context_length >= 1024 {
            format!("{}K", summary.context_length / 1024)
        } else {
            format!("{}", summary.context_length)
        };

        rows.push(BenchmarkTableRow {
            context_label,
            ttft_ms: format!("{:.2}", summary.avg_ttft_ms),
            prefill_tps: format!("{:.2}", summary.avg_prefill_speed_tps),
            decode_tps: format!("{:.2}", summary.avg_decode_speed_tps),
            total_ms: format!("{:.2}", summary.avg_total_time_ms),
        });
    }

    println!("Sequential Results:");
    println!("{}", Table::new(rows).to_string());

    // Concurrent results
    if let Some(ref cr) = concurrent_result {
        println!("\nConcurrent Results:");
        println!("  Requests: {}", cr.num_requests);
        println!("  Total Time: {:.2} ms", cr.total_time_ms);
        println!("  Avg Latency: {:.2} ms", cr.avg_time_ms);
        println!("  Min Latency: {:.2} ms", cr.min_time_ms);
        println!("  Max Latency: {:.2} ms", cr.max_time_ms);
        println!("  Throughput: {:.2} req/s", cr.throughput_rps);
    }
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
