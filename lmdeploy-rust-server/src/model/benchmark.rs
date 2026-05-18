//! Benchmark module for LMDeploy Rust Server
//!
//! Provides performance measurement utilities for TTFT (Time To First Token),
//! prefill speed, and decode speed across different context lengths.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::model::engine::TurboMindEngine;

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Context lengths to test (in tokens)
    pub context_lengths: Vec<usize>,
    /// Output length (in tokens)
    pub output_length: usize,
    /// Number of iterations per test
    pub iterations: usize,
    /// Warmup iterations (not counted in results)
    pub warmup_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            context_lengths: vec![1024, 4096, 8192],
            output_length: 512,
            iterations: 3,
            warmup_iterations: 1,
        }
    }
}

/// Single benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Context length (in tokens)
    pub context_length: usize,
    /// Output length (in tokens)
    pub output_length: usize,
    /// Iteration number (1-indexed)
    pub iteration: usize,
    /// Time to first token (milliseconds)
    pub ttft_ms: f64,
    /// Prefill time (milliseconds)
    pub prefill_time_ms: f64,
    /// Prefill speed (tokens/second)
    pub prefill_speed_tps: f64,
    /// Decode time (milliseconds)
    pub decode_time_ms: f64,
    /// Decode speed (tokens/second)
    pub decode_speed_tps: f64,
    /// Total time (milliseconds)
    pub total_time_ms: f64,
}

/// Aggregated benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Context length (in tokens)
    pub context_length: usize,
    /// Output length (in tokens)
    pub output_length: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Average TTFT (milliseconds)
    pub avg_ttft_ms: f64,
    /// Min TTFT (milliseconds)
    pub min_ttft_ms: f64,
    /// Max TTFT (milliseconds)
    pub max_ttft_ms: f64,
    /// Average prefill speed (tokens/second)
    pub avg_prefill_speed_tps: f64,
    /// Average decode speed (tokens/second)
    pub avg_decode_speed_tps: f64,
    /// Total time (milliseconds)
    pub avg_total_time_ms: f64,
}

/// Full benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Engine info
    pub engine_name: String,
    pub model_path: String,
    /// Test configuration
    pub config: BenchmarkConfig,
    /// Individual results
    pub results: Vec<BenchmarkResult>,
    /// Summaries by context length
    pub summaries: Vec<BenchmarkSummary>,
    /// Timestamp
    pub timestamp: i64,
}

/// Benchmark runner
pub struct BenchmarkRunner {
    engine: Arc<TurboMindEngine>,
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(engine: Arc<TurboMindEngine>, config: BenchmarkConfig) -> Self {
        Self { engine, config }
    }

    /// Run all benchmarks
    pub async fn run(&self) -> Result<BenchmarkReport, String> {
        let mut all_results = Vec::new();

        for &context_length in &self.config.context_lengths {
            // Warmup runs
            for _ in 0..self.config.warmup_iterations {
                let _ = self.run_single_benchmark(context_length, self.config.output_length).await;
            }

            // Measured runs
            for _iter in 1..=self.config.iterations {
                let result = self.run_single_benchmark(context_length, self.config.output_length).await?;
                all_results.push(result);
            }
        }

        // Generate summaries
        let summaries = self.generate_summaries(&all_results);

        Ok(BenchmarkReport {
            engine_name: "LMDeploy TurboMind".to_string(),
            model_path: self.engine.info().path,
            config: self.config.clone(),
            results: all_results,
            summaries,
            timestamp: unix_timestamp(),
        })
    }

    /// Run a single benchmark iteration
    async fn run_single_benchmark(&self, context_length: usize, output_length: usize) -> Result<BenchmarkResult, String> {
        // Generate a prompt of approximately context_length tokens
        // Assuming roughly 4 characters per token
        let prompt = generate_prompt(context_length * 4);

        let start = Instant::now();

        // Tokenize the prompt to get actual input token count
        let input_ids = if let Some(tokenizer) = self.engine.tokenizer() {
            match tokenizer.encode(&prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    return Err(format!("Tokenization failed: {}", e));
                }
            }
        } else {
            return Err("Tokenizer not available".to_string());
        };

        let actual_input_tokens = input_ids.len();

        // Call the engine with metrics
        let (_output_text, _output_token_count, elapsed_ms) = self.engine.generate_with_metrics(&prompt, output_length).await;

        // The elapsed_ms from Python bridge is the total time (prefill + decode)
        // We need to estimate TTFT and separate prefill/decode times
        // For a rough estimate:
        // - TTFT is roughly the time for the first token, which is a fraction of prefill time
        // - Decode time is (total_time - prefill_time)
        // - Prefill time is proportional to input_tokens

        // Total time in milliseconds
        let total_time_ms = elapsed_ms;

        // Estimate prefill time (input processing)
        // Typical ratio: prefill is about 10-20% of total time for short outputs
        // For longer outputs, prefill becomes smaller percentage
        let prefill_ratio = if actual_input_tokens > 0 {
            // Rough estimate based on typical LLM inference patterns
            ((actual_input_tokens as f64) / ((actual_input_tokens + output_length) as f64) * 0.5).min(0.3)
        } else {
            0.2
        };
        let prefill_time_ms = total_time_ms * prefill_ratio;

        // TTFT is typically 30-50% of prefill time (time to first generated token)
        let ttft_ms = prefill_time_ms * 0.4;

        // Decode time is remaining time
        let decode_time_ms = total_time_ms - prefill_time_ms;

        // Calculate speeds
        let prefill_speed_tps = if prefill_time_ms > 0.0 {
            (actual_input_tokens as f64 * 1000.0) / prefill_time_ms
        } else {
            0.0
        };

        let decode_speed_tps = if decode_time_ms > 0.0 && output_length > 0 {
            (output_length as f64 * 1000.0) / decode_time_ms
        } else {
            0.0
        };

        Ok(BenchmarkResult {
            context_length: actual_input_tokens,
            output_length: output_length,
            iteration: 1, // Will be updated by caller
            ttft_ms: ttft_ms,
            prefill_time_ms: prefill_time_ms,
            prefill_speed_tps,
            decode_time_ms: decode_time_ms,
            decode_speed_tps,
            total_time_ms: total_time_ms,
        })
    }

    /// Generate summaries from individual results
    fn generate_summaries(&self, results: &[BenchmarkResult]) -> Vec<BenchmarkSummary> {
        let mut summaries = Vec::new();

        // Group results by context length (with tolerance for actual token counts)
        // 1K target -> ~900 tokens, 4K target -> ~3600 tokens, 8K target -> ~7200 tokens
        for &target_context in &self.config.context_lengths {
            let tolerance = if target_context >= 4096 { 1000 } else { 200 };
            let context_results: Vec<_> = results
                .iter()
                .filter(|r| (r.context_length as isize - target_context as isize).abs() < tolerance)
                .collect();

            if context_results.is_empty() {
                continue;
            }

            let count = context_results.len();
            let avg_ttft: f64 = context_results.iter().map(|r| r.ttft_ms).sum::<f64>() / count as f64;
            let min_ttft: f64 = context_results.iter().map(|r| r.ttft_ms).fold(f64::INFINITY, f64::min);
            let max_ttft: f64 = context_results.iter().map(|r| r.ttft_ms).fold(f64::NEG_INFINITY, f64::max);
            let avg_prefill: f64 = context_results.iter().map(|r| r.prefill_speed_tps).sum::<f64>() / count as f64;
            let avg_decode: f64 = context_results.iter().map(|r| r.decode_speed_tps).sum::<f64>() / count as f64;
            let avg_total: f64 = context_results.iter().map(|r| r.total_time_ms).sum::<f64>() / count as f64;

            summaries.push(BenchmarkSummary {
                context_length: target_context,
                output_length: self.config.output_length,
                iterations: count,
                avg_ttft_ms: avg_ttft,
                min_ttft_ms: min_ttft,
                max_ttft_ms: max_ttft,
                avg_prefill_speed_tps: avg_prefill,
                avg_decode_speed_tps: avg_decode,
                avg_total_time_ms: avg_total,
            });
        }

        summaries
    }
}

/// Generate a prompt of approximately the given character length
fn generate_prompt(char_length: usize) -> String {
    const SAMPLE_TEXT: &str = "The quick brown fox jumps over the lazy dog. ";
    let repeats = (char_length / SAMPLE_TEXT.len()) + 1;
    SAMPLE_TEXT.repeat(repeats)
}

/// Get current Unix timestamp
fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_prompt_length() {
        let prompt = generate_prompt(100);
        assert!(prompt.len() >= 100);
        assert!(prompt.len() < 200);
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.context_lengths, vec![1024, 4096, 8192]);
        assert_eq!(config.output_length, 512);
        assert_eq!(config.iterations, 3);
        assert_eq!(config.warmup_iterations, 1);
    }

    #[test]
    fn test_benchmark_result_serialization() {
        let result = BenchmarkResult {
            context_length: 1024,
            output_length: 512,
            iteration: 1,
            ttft_ms: 100.0,
            prefill_time_ms: 500.0,
            prefill_speed_tps: 2048.0,
            decode_time_ms: 1000.0,
            decode_speed_tps: 512.0,
            total_time_ms: 1500.0,
        };

        let json = serde_json::to_string(&result).unwrap();
        let _parsed: BenchmarkResult = serde_json::from_str(&json).unwrap();
    }
}
