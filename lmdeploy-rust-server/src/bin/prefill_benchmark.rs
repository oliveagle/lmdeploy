/// Rust Server Prefill Benchmark
///
/// Measures prefill performance of the Rust server using the same methodology
/// as the Python TurboMind benchmark. Directly calls TurboMindCEngine (no gRPC
/// overhead). Uses UNIFIED TESTING PARAMETERS:
///   - Input lengths: 512, 1024, 4096, 8192
///   - Output length: 512 (for decode testing)
///   - Warmup runs: 2
///   - Measure runs: 5
///   - Concurrency: 1
///
/// Usage:
///     cargo run --bin prefill_benchmark [--model /path/to/model] [--output tests/prefill_benchmark_rust.json]

use std::time::Instant;
use clap::Parser;

use lmdeploy_server::model::cpp_engine::{GenerationParams, TurboMindCEngine};

/// Benchmark output module
mod bench;

/// Test context configurations (label, target token count)
/// Matches Python TurboMind benchmark exactly: 512, 1024, 4096, 8192
const TEST_CONTEXTS: &[(&str, usize)] = &[
    ("512", 512),
    ("1K", 1024),
    ("4K", 4096),
    ("8K", 8192),
];

const REPEAT_TEXT: &str = "The quick brown fox jumps over the lazy dog. ";
const SEPARATOR: &str = "================================================================================";

#[derive(Parser, Debug)]
#[command(name = "prefill_benchmark")]
#[command(about = "Benchmark Rust server prefill performance")]
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

/// Generate a prompt with approximately target_token_count tokens
fn gen_prompt(target_token_count: usize) -> String {
    let chars_per_token = 4;
    let target_chars = target_token_count * chars_per_token;
    let repeats = (target_chars / REPEAT_TEXT.len()) + 1;
    REPEAT_TEXT.repeat(repeats)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let model_path = args.model.unwrap_or_else(|| {
        "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".to_string()
    });

    println!("{SEPARATOR}");
    println!("Rust Server Prefill 性能测试");
    println!("{SEPARATOR}");
    println!();
    println!("Model:   {model_path}");
    println!("Warmup:  {}, Measure: {}", args.warmup, args.measure);

    // Create engine
    println!();
    println!("[1/3] 加载模型...");
    let engine = TurboMindCEngine::new(&model_path).await?;
    println!("模型加载成功");

    let tokenizer = engine.tokenizer().ok_or("Tokenizer not available")?;

    // Warmup
    println!();
    println!("[2/3] Warmup ({} runs)...", args.warmup);
    let warmup_prompt = gen_prompt(2048);
    for i in 0..args.warmup {
        let params = default_gen_params(1);
        print!("  warmup {}... ", i + 1);
        engine.generate(&warmup_prompt, params).await;
        println!("OK");
    }

    // Measure prefill performance
    println!();
    println!("[3/3] 测量预填充性能 ({} runs)...", args.measure);
    println!();
    println!(
        "{:>8} | {:>8} | {:>10} | {:>10} | {:>12} | {:>12}",
        "Context", "Tokens", "Avg TTFT", "Min TTFT", "Avg TPS", "Max TPS"
    );
    println!("{SEPARATOR}");

    let mut results: std::collections::HashMap<String, bench::ContextResult> =
        std::collections::HashMap::new();

    for &(label, target_tokens) in TEST_CONTEXTS {
        let prompt = gen_prompt(target_tokens);

        let actual_tokens = tokenizer
            .encode(&prompt, false, false)
            .map(|ids| ids.len())
            .unwrap_or(target_tokens);

        let mut run_times: Vec<f64> = Vec::new();

        print!("{:>8} ({:>6} tok)... ", label, actual_tokens);

        for _r in 0..args.measure {
            let params = default_gen_params(1);
            let start = Instant::now();
            engine.generate(&prompt, params).await;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            run_times.push(elapsed_ms);
            print!("{:.1}ms ", elapsed_ms);
        }

        if run_times.len() == args.measure {
            let avg_ms = run_times.iter().sum::<f64>() / run_times.len() as f64;
            let min_ms = run_times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_ms = run_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg_tps = actual_tokens as f64 / avg_ms * 1000.0;
            let max_tps = actual_tokens as f64 / min_ms * 1000.0;

            results.insert(
                label.to_string(),
                bench::ContextResult {
                    ctx_tokens: actual_tokens,
                    avg_ms,
                    min_ms,
                    max_ms,
                    avg_tps,
                    max_tps,
                },
            );

            println!(
                "| Avg {:>8.1} tok/s, Max {:>8.1} tok/s (TTFT: {:.1}ms)",
                avg_tps, max_tps, avg_ms
            );
        } else {
            println!("| ERROR - insufficient results");
        }
    }

    // Summary
    println!();
    println!("{SEPARATOR}");
    println!("总结 - Rust Server Prefill 性能");
    println!("{SEPARATOR}");

    for &label in &["512", "1K", "4K", "8K"] {
        if let Some(r) = results.get(label) {
            println!("  {label:>6}: {:>10.1} tok/s (TTFT: {:>8.1}ms)", r.avg_tps, r.avg_ms);
        }
    }

    // Save results
    let output = bench::BenchmarkResult {
        model: model_path.clone(),
        config: bench::BenchmarkConfig {
            warmup_runs: args.warmup,
            measure_runs: args.measure,
            engine: "pure_cpp".to_string(),
        },
        results,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64(),
    };

    let output_path = args.output.unwrap_or_else(|| "tests/prefill_benchmark_rust.json".to_string());
    bench::save_benchmark(&output_path, &output)?;

    println!();
    println!("结果已保存: {output_path}");

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
