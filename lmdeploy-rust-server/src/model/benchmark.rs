//! Benchmark module for LMDeploy Rust Server
//!
//! Provides performance measurement utilities for TTFT (Time To First Token),
//! prefill speed, and decode speed across different context lengths.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::model::cpp_engine::TurboMindCEngine;
use crate::model::GenerationParams;
use futures::StreamExt;

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

/// Single benchmark result (with actual streaming measurements)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Context length (in tokens)
    pub context_length: usize,
    /// Output length (in tokens)
    pub output_length: usize,
    /// Iteration number (1-indexed)
    pub iteration: usize,
    /// Time to first token (milliseconds) - actual measured via streaming
    pub ttft_ms: f64,
    /// Prefill time (milliseconds) - actual measured (time to first token - prefill computation)
    pub prefill_time_ms: f64,
    /// Prefill speed (tokens/second)
    pub prefill_speed_tps: f64,
    /// Decode time (milliseconds)
    pub decode_time_ms: f64,
    /// Decode speed (tokens/second)
    pub decode_speed_tps: f64,
    /// Total time (milliseconds)
    pub total_time_ms: f64,
    /// Actual output tokens generated
    pub actual_output_tokens: usize,
    /// Individual inter-token latencies (milliseconds)
    pub itl_ms: Vec<f64>,
}

/// Streaming benchmark result (with per-token timing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingBenchmarkResult {
    /// Context length (in tokens)
    pub context_length: usize,
    /// Output length requested (in tokens)
    pub output_length: usize,
    /// Actual output tokens generated
    pub actual_output_tokens: usize,
    /// Iteration number
    pub iteration: usize,
    /// Time to first token (milliseconds) - actual measured
    pub ttft_ms: f64,
    /// Total time (milliseconds)
    pub total_time_ms: f64,
    /// Decode time (total - TTFT, in milliseconds)
    pub decode_time_ms: f64,
    /// Decode speed (tokens/second)
    pub decode_speed_tps: f64,
    /// Average inter-token latency (milliseconds)
    pub avg_itl_ms: f64,
    /// Min inter-token latency (milliseconds)
    pub min_itl_ms: f64,
    /// Max inter-token latency (milliseconds)
    pub max_itl_ms: f64,
    /// P50 inter-token latency (milliseconds)
    pub p50_itl_ms: f64,
    /// P95 inter-token latency (milliseconds)
    pub p95_itl_ms: f64,
    /// P99 inter-token latency (milliseconds)
    pub p99_itl_ms: f64,
    /// Individual token timestamps (milliseconds from start)
    pub token_timestamps_ms: Vec<f64>,
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

/// Benchmark runner with streaming support
pub struct BenchmarkRunner {
    engine: Arc<TurboMindCEngine>,
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(engine: Arc<TurboMindCEngine>, config: BenchmarkConfig) -> Self {
        Self { engine, config }
    }

    /// Run all benchmarks using streaming for accurate TTFT measurement
    pub async fn run(&self) -> Result<BenchmarkReport, String> {
        let mut all_results = Vec::new();

        for &context_length in &self.config.context_lengths {
            // Warmup runs
            for _ in 0..self.config.warmup_iterations {
                let _ = self.run_streaming_benchmark(context_length, self.config.output_length, 0).await;
            }

            // Measured runs
            for iter in 1..=self.config.iterations {
                let result = self.run_streaming_benchmark(context_length, self.config.output_length, iter).await?;
                all_results.push(result);
            }
        }

        // Generate summaries
        let summaries = self.generate_summaries(&all_results);

        Ok(BenchmarkReport {
            engine_name: "LMDeploy TurboMind C++ (streaming)".to_string(),
            model_path: self.engine.model_path.clone(),
            config: self.config.clone(),
            results: all_results,
            summaries,
            timestamp: unix_timestamp(),
        })
    }

    /// Run a single benchmark iteration using streaming for accurate timing
    async fn run_streaming_benchmark(&self, context_length: usize, output_length: usize, iteration: usize) -> Result<BenchmarkResult, String> {
        // Generate a prompt of approximately context_length tokens
        let prompt = generate_prompt(context_length * 4);

        // Tokenize to get actual input token count
        let input_ids = match self.engine.tokenizer() {
            Some(t) => match t.encode(&prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => return Err(format!("Tokenization failed: {}", e)),
            },
            None => return Err("Tokenizer not available".to_string()),
        };
        let actual_input_tokens = input_ids.len();

        // Use streaming to get actual TTFT and per-token timing
        let params = GenerationParams {
            max_tokens: Some(output_length),
            ..Default::default()
        };

        let start = Instant::now();
        let mut stream = self.engine.generate_stream(&prompt, params).await;

        let mut ttft_ms = 0.0;
        let mut first_token_received = false;
        let mut total_tokens = 0;
        let mut last_token_time = 0.0;
        let mut token_times: Vec<f64> = Vec::new();

        // Collect tokens from stream
        while let Some(_token) = stream.next().await {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms

            if !first_token_received {
                ttft_ms = elapsed;
                first_token_received = true;
            }
            last_token_time = elapsed;
            token_times.push(elapsed);
            total_tokens += 1;
        }

        let total_time_ms = last_token_time;

        // Calculate inter-token latencies
        let mut itl_ms: Vec<f64> = Vec::new();
        for i in 1..token_times.len() {
            itl_ms.push(token_times[i] - token_times[i-1]);
        }

        // Calculate prefill time (TTFT for the first output token)
        let prefill_time_ms = ttft_ms;

        // Decode time is remaining time after first token
        let decode_time_ms = total_time_ms - ttft_ms;

        // Calculate prefill speed (input processing rate)
        let prefill_speed_tps = if prefill_time_ms > 0.0 {
            (actual_input_tokens as f64 * 1000.0) / prefill_time_ms
        } else {
            0.0
        };

        // Calculate decode speed (output processing rate)
        let decode_speed_tps = if decode_time_ms > 0.0 && total_tokens > 0 {
            (total_tokens as f64 * 1000.0) / decode_time_ms
        } else {
            0.0
        };

        Ok(BenchmarkResult {
            context_length: actual_input_tokens,
            output_length: output_length,
            iteration,
            ttft_ms,
            prefill_time_ms,
            prefill_speed_tps,
            decode_time_ms,
            decode_speed_tps,
            total_time_ms,
            actual_output_tokens: total_tokens,
            itl_ms,
        })
    }

    /// Generate summaries from individual results
    fn generate_summaries(&self, results: &[BenchmarkResult]) -> Vec<BenchmarkSummary> {
        let mut summaries = Vec::new();

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
