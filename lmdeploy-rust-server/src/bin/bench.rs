//! Benchmark data structures and utilities
//!
//! Shared data structures for representing benchmark results,
//! compatible with the Python benchmark JSON schema.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Benchmark configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup runs
    pub warmup_runs: usize,
    /// Number of measurement runs
    pub measure_runs: usize,
    /// Engine type identifier
    pub engine: String,
}

/// Result for a single context length measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResult {
    /// Actual token count of the input prompt
    pub ctx_tokens: usize,
    /// Average time-to-first-token in milliseconds
    pub avg_ms: f64,
    /// Minimum time-to-first-token in milliseconds
    pub min_ms: f64,
    /// Maximum time-to-first-token in milliseconds
    pub max_ms: f64,
    /// Average tokens-per-second rate
    pub avg_tps: f64,
    /// Maximum tokens-per-second rate
    pub max_tps: f64,
}

/// Complete benchmark result with all context lengths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Model path
    pub model: String,
    /// Benchmark configuration
    pub config: BenchmarkConfig,
    /// Results per context (keys: "1K", "2K", "4K", "8K")
    pub results: HashMap<String, ContextResult>,
    /// Timestamp when measurement was taken
    pub timestamp: f64,
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self {
            model: String::new(),
            config: BenchmarkConfig {
                warmup_runs: 2,
                measure_runs: 5,
                engine: "pure_cpp".to_string(),
            },
            results: HashMap::new(),
            timestamp: 0.0,
        }
    }
}

/// Load a benchmark result from a JSON file
pub fn load_benchmark(path: &str) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let result: BenchmarkResult = serde_json::from_str(&content)?;
    Ok(result)
}

/// Save a benchmark result to a JSON file
pub fn save_benchmark(path: &str, result: &BenchmarkResult) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Compare two benchmark results and print a summary
pub fn compare_benchmarks(a: &BenchmarkResult, b: &BenchmarkResult, label_a: &str, label_b: &str) {
    println!();
    println!("{}", "=".repeat(80));
    println!("Benchmark Comparison: {} vs {}", label_a, label_b);
    println!("{}", "=".repeat(80));
    println!();

    let contexts = vec!["1K", "2K", "4K", "8K"];
    println!("{:>8} | {:>15} {:>15} {:>10}",
        "Context", label_a, label_b, "Diff %");
    println!("{}", "-".repeat(80));

    for ctx in contexts {
        match (a.results.get(ctx), b.results.get(ctx)) {
            (Some(r_a), Some(r_b)) => {
                let diff = (r_b.avg_tps - r_a.avg_tps) / r_a.avg_tps * 100.0;
                println!("{:>8} | {:>12.1} tok/s {:>12.1} tok/s {:>+9.1}%",
                    ctx, r_a.avg_tps, r_b.avg_tps, diff);
            }
            (Some(r_a), None) => {
                println!("{:>8} | {:>12.1} tok/s {:>12} {:>10}",
                    ctx, r_a.avg_tps, "N/A", "");
            }
            (None, Some(r_b)) => {
                println!("{:>8} | {:>12} {:>12.1} tok/s {:>10}",
                    ctx, "N/A", r_b.avg_tps, "");
            }
            (None, None) => {
                println!("{:>8} | {:>12} {:>12} {:>10}",
                    ctx, "N/A", "N/A", "");
            }
        }
    }
}
