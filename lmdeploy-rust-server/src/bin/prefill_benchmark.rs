//! Rust Server Prefill/Decode Benchmark
//!
//! 统一测试方法 - 与 Python TurboMind benchmark 完全一致：
//! - 输入长度: 512, 1024, 2048, 4096, 8192 (与 Python 完全相同)
//! - 输出长度: 512 tokens (与 Python 相同)
//! - 预热: 2 次，测量: 5 次 (与 Python 相同)
//! - 并发度: 1 (串行测试，与 Python 相同)
//! - 计算公式: Prefill (tok/s) = input_len / (ttft_ms / 1000)
//!           Decode (tok/s) = 1000 / tpot_ms
//!
//! Usage:
//!     cargo run --bin prefill_benchmark [--model /path/to/model] [--output results.json]

use std::sync::Arc;
use std::time::Instant;
use clap::Parser;
use futures::StreamExt;

use lmdeploy_server::model::cpp_engine::{GenerationParams, TurboMindCEngine};

/// Test context configurations (label, target token count)
/// Matches Python TurboMind benchmark exactly: 512, 1024, 2048, 4096, 8192
const TEST_CONTEXTS: &[(&str, usize)] = &[
    ("512", 512),
    ("1K", 1024),
    ("2K", 2048),
    ("4K", 4096),
    ("8K", 8192),
];

const REPEAT_TEXT: &str = "The quick brown fox jumps over the lazy dog. ";
const OUTPUT_LENGTH: usize = 512;

#[derive(Parser, Debug)]
#[command(name = "prefill_benchmark")]
#[command(about = "Unified prefill/decode benchmark matching Python methodology")]
struct Args {
    /// Model path
    #[arg(long)]
    model: Option<String>,

    /// Output JSON file path
    #[arg(long)]
    output: Option<String>,

    /// Number of warmup runs
    #[arg(long, default_value = "2")]
    warmup: usize,

    /// Number of measurement runs per context
    #[arg(long, default_value = "5")]
    measure: usize,
}

/// Benchmark result for a single iteration
#[derive(Debug, Clone, serde::Serialize)]
struct IterationResult {
    iteration: usize,
    input_len: usize,
    ttft_ms: f64,
    total_time_ms: f64,
    prefill_tps: f64,
    decode_tps: f64,
}

/// Summary for a context length
#[derive(Debug, Clone, serde::Serialize)]
struct ContextSummary {
    input_len: usize,
    output_len: usize,
    iterations: usize,
    ttft_ms_avg: f64,
    ttft_ms_p99: f64,
    prefill_tps: f64,
    decode_tps: f64,
    total_time_ms_avg: f64,
}

/// Full benchmark report
#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkReport {
    engine: String,
    model: String,
    backend: String,
    date: String,
    input_lengths: Vec<usize>,
    output_length: usize,
    warmup_runs: usize,
    measure_runs: usize,
    summaries: Vec<ContextSummary>,
}

/// Generate a prompt with approximately target_token_count tokens
fn gen_prompt(target_token_count: usize) -> String {
    let chars_per_token = 4;
    let target_chars = target_token_count * chars_per_token;
    let repeats = (target_chars / REPEAT_TEXT.len()) + 1;
    REPEAT_TEXT.repeat(repeats)
}

/// Run a single iteration using streaming for accurate TTFT measurement
async fn run_iteration(
    engine: &TurboMindCEngine,
    input_len: usize,
    output_len: usize,
    iteration: usize,
) -> Result<IterationResult, String> {
    let prompt = gen_prompt(input_len);

    let params = GenerationParams {
        max_tokens: Some(output_len),
        temperature: Some(0.0),  // Deterministic output
        top_p: Some(1.0),
        ..Default::default()
    };

    let start = Instant::now();
    let mut stream = engine.generate_stream(&prompt, params).await;
    futures::pin_mut!(stream);

    let mut ttft_ms = 0.0;
    let mut first_token_received = false;
    let mut last_token_time = 0.0;
    let mut token_count = 0;

    while let Some((_, _)) = stream.next().await {
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        if !first_token_received {
            ttft_ms = elapsed;
            first_token_received = true;
        }

        last_token_time = elapsed;
        token_count += 1;
    }

    let total_time_ms = last_token_time;
    let decode_time_ms = total_time_ms - ttft_ms;

    // Calculate throughput: same formula as Python
    // Prefill (tok/s) = input_len / (ttft_ms / 1000)
    let prefill_tps = if ttft_ms > 0.0 {
        input_len as f64 / (ttft_ms / 1000.0)
    } else {
        0.0
    };

    // Decode (tok/s) = 1000 / tpot_ms, where tpot_ms = decode_time / output_tokens
    let decode_tps = if token_count > 0 && decode_time_ms > 0.0 {
        let tpot_ms = decode_time_ms / token_count as f64;
        1000.0 / tpot_ms
    } else {
        0.0
    };

    Ok(IterationResult {
        iteration,
        input_len,
        ttft_ms,
        total_time_ms,
        prefill_tps,
        decode_tps,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let model_path = args.model.unwrap_or_else(|| {
        "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".to_string()
    });

    println!("\n{}", "=".repeat(80));
    println!("LMDeploy Rust Server Benchmark (统一测试方法)");
    println!("{}", "=".repeat(80));
    println!("Model:   {}", model_path);
    println!("Backend: turbomind (C++)");
    println!("Input:   512, 1024, 2048, 4096, 8192 (与 Python 完全一致)");
    println!("Output:  512 tokens (与 Python 一致)");
    println!("Warmup:  {} runs", args.warmup);
    println!("Measure:  {} runs per context", args.measure);
    println!("Concurrency: 1 (串行测试)");
    println!("{}", "=".repeat(80));

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Load engine
    println!("\n[1/3] Loading TurboMind engine...");
    let engine = Arc::new(TurboMindCEngine::new(&model_path).await?);
    println!("Engine loaded successfully");

    // Warmup
    println!("\n[2/3] Warmup ({} runs)...", args.warmup);
    let warmup_prompt = gen_prompt(2048);
    for i in 0..args.warmup {
        let params = default_gen_params(1);
        print!("  warmup {}... ", i + 1);
        engine.generate(&warmup_prompt, params).await;
        println!("OK");
    }

    // Measure performance
    println!("\n[3/3] Measuring Prefill/Decode performance...");
    println!();
    println!(
        "{:<8} | {:>10} | {:>10} | {:>12} | {:>12}",
        "Context", "TTFT (ms)", "Prefill (tok/s)", "Decode (tok/s)", "Total (ms)"
    );
    println!("{}", "-".repeat(80));

    let mut all_results: Vec<IterationResult> = Vec::new();

    for &(label, target_tokens) in TEST_CONTEXTS {
        let input_len = target_tokens;
        let output_len = OUTPUT_LENGTH;

        print!("{:<8} (input={:>6}, out={:>5})... ", label, input_len, output_len);

        let mut context_results = Vec::new();

        for iter in 1..=args.measure {
            match run_iteration(&engine, input_len, output_len, iter).await {
                Ok(result) => {
                    context_results.push(result.clone());
                    all_results.push(result.clone());
                    print!(
                        "{:6.1} ",
                        result.ttft_ms
                    );
                }
                Err(e) => {
                    eprintln!("| ERROR: {}", e);
                }
            }
        }

        if !context_results.is_empty() {
            let avg_ttft: f64 = context_results.iter().map(|r| r.ttft_ms).sum::<f64>()
                / context_results.len() as f64;
            let avg_prefill: f64 = context_results.iter().map(|r| r.prefill_tps).sum::<f64>()
                / context_results.len() as f64;
            let avg_decode: f64 = context_results.iter().map(|r| r.decode_tps).sum::<f64>()
                / context_results.len() as f64;

            let mut sorted_ttfts: Vec<f64> = context_results.iter().map(|r| r.ttft_ms).collect();
            sorted_ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ttft_p99 = if sorted_ttfts.len() > 1 {
                let idx = (sorted_ttfts.len() as f64 * 0.99) as usize;
                *sorted_ttfts.get(idx).unwrap_or(&0.0)
            } else {
                0.0
            };

            println!(
                "| Avg TTFT: {:7.2}ms (P99: {:7.2}ms), Prefill: {:>10.1} tok/s, Decode: {:>10.1} tok/s",
                avg_ttft, ttft_p99, avg_prefill, avg_decode
            );
        }
    }

    // Generate summary report
    let mut summaries = Vec::new();
    for &input_len in &[512, 1024, 2048, 4096, 8192] {
        let context_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.input_len == input_len)
            .cloned()
            .collect();

        if !context_results.is_empty() {
            let avg_ttft = context_results.iter().map(|r| r.ttft_ms).sum::<f64>()
                / context_results.len() as f64;
            let avg_prefill = context_results.iter().map(|r| r.prefill_tps).sum::<f64>()
                / context_results.len() as f64;
            let avg_decode = context_results.iter().map(|r| r.decode_tps).sum::<f64>()
                / context_results.len() as f64;
            let avg_total = context_results.iter().map(|r| r.total_time_ms).sum::<f64>()
                / context_results.len() as f64;

            summaries.push(ContextSummary {
                input_len,
                output_len: OUTPUT_LENGTH,
                iterations: context_results.len(),
                ttft_ms_avg: avg_ttft,
                ttft_ms_p99: 0.0, // Placeholder
                prefill_tps: avg_prefill,
                decode_tps: avg_decode,
                total_time_ms_avg: avg_total,
            });
        }
    }

    // Create report
    let report = BenchmarkReport {
        engine: "LMDeploy TurboMind C++".to_string(),
        model: model_path.clone(),
        backend: "turbomind".to_string(),
        date: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        input_lengths: vec![512, 1024, 2048, 4096, 8192],
        output_length: OUTPUT_LENGTH,
        warmup_runs: args.warmup,
        measure_runs: args.measure,
        summaries,
    };

    // Save to JSON
    let output_path = args.output.unwrap_or_else(|| {
        format!("results/prefill_benchmark_rust_{}.json", chrono::Utc::now().format("%Y%m%d_%H%M%S"))
    });

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&output_path, json)?;

    println!("\n{}", "=".repeat(80));
    println!("Results saved to: {}", output_path);
    println!("{}", "=".repeat(80));

    Ok(())
}

fn default_gen_params(max_tokens: usize) -> GenerationParams {
    GenerationParams {
        max_tokens: Some(max_tokens),
        temperature: Some(0.7),
        top_p: Some(0.95),
        top_k: Some(50),
        min_p: None,
        repetition_penalty: None,
        seed: None,
        stop: None,
        logprobs: None,
        top_logprobs: None,
        grammar: None,
    }
}
