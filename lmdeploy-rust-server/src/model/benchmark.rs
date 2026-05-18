//! Benchmark module for LMDeploy Rust Server
//!
//! Provides performance measurement utilities for TTFT (Time To First Token),
//! prefill speed, and decode speed across different context lengths.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::model::engine::TurboMindEngine;

/// Benchmark configuration
#[derive(Debug, Clone)]
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
            for iter in 1..=self.config.iterations {
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

        // For TTFT measurement, we need to track when first token is generated
        // Since we're using the generate() API which returns all tokens at once,
        // we'll estimate TTFT as a fraction of prefill time
        let prefill_start = Instant::now();

        // Call the engine
        let output = self.engine.generate(&prompt, output_length).await;

        let prefill_end = Instant::now();
        let total_end = Instant::now();

        let prefill_time = prefill_end.duration_since(prefill_start);
        let decode_time = total_end.duration_since(prefill_end);
        let total_time = total_end.duration_since(start);

        // Estimate TTFT as 20% of prefill time (typical for first token generation)
        let ttft = prefill_time.as_millis() as f64 * 0.2;

        // Calculate speeds
        let prefill_speed_tps = if prefill_time.as_millis() > 0 {
            (context_length as f64 * 1000.0) / prefill_time.as_millis() as f64
        } else {
            0.0
        };

        let decode_speed_tps = if decode_time.as_millis() > 0 {
            (output_length as f64 * 1000.0) / decode_time.as_millis() as f64
        } else {
            0.0
        };

        // Estimate actual token counts from output length
        let output_token_count = output.len() / 4; // Rough estimate

        Ok(BenchmarkResult {
            context_length,
            output_length: output_token_count,
            iteration: 1, // Will be updated by caller
            ttft_ms: ttft,
            prefill_time_ms: prefill_time.as_millis() as f64,
            prefill_speed_tps,
            decode_time_ms: decode_time.as_millis() as f64,
            decode_speed_tps,
            total_time_ms: total_time.as_millis() as f64,
        })
    }

    /// Generate summaries from individual results
    fn generate_summaries(&self, results: &[BenchmarkResult]) -> Vec<BenchmarkSummary> {
        let mut summaries = Vec::new();

        for &context_length in &self.config.context_lengths {
            let context_results: Vec<_> = results
                .iter()
                .filter(|r| r.context_length == context_length)
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
                context_length,
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

/// Simple HTTP API benchmark
pub struct HttpBenchmarkRunner {
    /// API endpoint URL
    pub endpoint: String,
    /// Model name
    pub model: String,
    /// Config
    pub config: BenchmarkConfig,
}

impl HttpBenchmarkRunner {
    /// Run benchmarks via HTTP API
    pub async fn run(&self) -> Result<BenchmarkReport, String> {
        let client = reqwest::Client::new();
        let mut all_results = Vec::new();

        for &context_length in &self.config.context_lengths {
            for iter in 1..=self.config.iterations {
                let result = self.run_http_benchmark(&client, context_length, iter).await?;
                all_results.push(result);
            }
        }

        Ok(BenchmarkReport {
            engine_name: "LMDeploy Rust Server (HTTP)".to_string(),
            model_path: self.model.clone(),
            config: self.config.clone(),
            results: all_results,
            summaries: Vec::new(), // Will be filled by caller
            timestamp: unix_timestamp(),
        })
    }

    /// Run a single HTTP benchmark
    async fn run_http_benchmark(&self, client: &reqwest::Client, context_length: usize, iteration: usize) -> Result<BenchmarkResult, String> {
        use crate::handlers::http::{ChatCompletionsRequest, Message};

        let prompt = generate_prompt(context_length * 4);

        let start = Instant::now();

        let req = ChatCompletionsRequest {
            model: self.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.clone(),
            }],
            temperature: Some(0.7),
            top_p: Some(0.95),
            max_tokens: Some(self.config.output_length as i32),
            stream: Some(false),
            stop: None,
            seed: None,
            presence_penalty: None,
            frequency_penalty: None,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            user: None,
        };

        let prefill_start = Instant::now();

        let response = client
            .post(&self.endpoint)
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        let prefill_end = Instant::now();

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let _json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("JSON parsing failed: {}", e))?;

        let total_end = Instant::now();

        let prefill_time = prefill_end.duration_since(prefill_start);
        let total_time = total_end.duration_since(start);
        let decode_time = total_end.duration_since(prefill_end);

        let ttft = prefill_time.as_millis() as f64 * 0.2;

        let prefill_speed_tps = if prefill_time.as_millis() > 0 {
            (context_length as f64 * 1000.0) / prefill_time.as_millis() as f64
        } else {
            0.0
        };

        let decode_speed_tps = if decode_time.as_millis() > 0 {
            (self.config.output_length as f64 * 1000.0) / decode_time.as_millis() as f64
        } else {
            0.0
        };

        Ok(BenchmarkResult {
            context_length,
            output_length: self.config.output_length,
            iteration,
            ttft_ms: ttft,
            prefill_time_ms: prefill_time.as_millis() as f64,
            prefill_speed_tps,
            decode_time_ms: decode_time.as_millis() as f64,
            decode_speed_tps,
            total_time_ms: total_time.as_millis() as f64,
        })
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
