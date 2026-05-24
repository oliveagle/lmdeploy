//! Benchmark module for LMDeploy Rust Server
//!
//! Provides performance measurement utilities for TTFT (Time To First Token),
//! prefill speed, and decode speed across different context lengths.
//! Also supports batch throughput benchmarking to verify vectorized inference.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::model::cpp_engine::{BatchItem, TurboMindCEngine};
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
            context_lengths: vec![1024, 4096, 8192, 16384, 32768, 49152, 65536, 131072],
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
    /// Tokenization time (milliseconds) - time to encode prompt to tokens
    pub tokenization_time_ms: f64,
    /// Pool acquisition time (milliseconds) - time to acquire request slot
    pub pool_acquire_time_ms: f64,
    /// Pure C++ engine time (milliseconds) - TTFT minus tokenization and pool time
    pub engine_time_ms: f64,
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
                let _ = self
                    .run_streaming_benchmark(context_length, self.config.output_length, 0)
                    .await;
            }

            // Measured runs
            for iter in 1..=self.config.iterations {
                let result = self
                    .run_streaming_benchmark(context_length, self.config.output_length, iter)
                    .await?;
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
    async fn run_streaming_benchmark(
        &self,
        context_length: usize,
        output_length: usize,
        iteration: usize,
    ) -> Result<BenchmarkResult, String> {
        // Generate a prompt of approximately context_length tokens
        let prompt = generate_prompt(context_length * 4);

        // Phase 1: Tokenize and measure time
        let tok_start = Instant::now();
        let input_ids = match self.engine.tokenizer() {
            Some(t) => match t.encode(&prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => return Err(format!("Tokenization failed: {}", e)),
            },
            None => return Err("Tokenizer not available".to_string()),
        };
        let tokenization_time_ms = tok_start.elapsed().as_secs_f64() * 1000.0;
        let actual_input_tokens = input_ids.len();

        // Phase 2: Pool acquisition - measure time to acquire a request slot
        let pool = match self.engine.pool() {
            Some(p) => p,
            None => return Err("Pool not available".to_string()),
        };
        let pool_start = Instant::now();
        let (_permit, mut _request, mut _input_tensors, mut _output_tensors) =
            pool.acquire().await;
        let pool_acquire_time_ms = pool_start.elapsed().as_secs_f64() * 1000.0;

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
            itl_ms.push(token_times[i] - token_times[i - 1]);
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

        // Pure C++ engine time = TTFT - pool_acquire_time_ms
        // (tokenization is done before the main flow, so subtract from TTFT)
        let engine_time_ms = ttft_ms - pool_acquire_time_ms;

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
            tokenization_time_ms,
            pool_acquire_time_ms,
            engine_time_ms,
        })
    }

    /// Generate summaries from individual results
    fn generate_summaries(&self, results: &[BenchmarkResult]) -> Vec<BenchmarkSummary> {
        let mut summaries = Vec::new();

        for &target_context in &self.config.context_lengths {
            let tolerance = match target_context {
                0..=4096 => 200,
                4097..=16384 => 1000,
                16385..=65536 => 3000,
                _ => 5000,
            };
            let context_results: Vec<_> = results
                .iter()
                .filter(|r| (r.context_length as isize - target_context as isize).abs() < tolerance)
                .collect();

            if context_results.is_empty() {
                continue;
            }

            let count = context_results.len();
            let avg_ttft: f64 =
                context_results.iter().map(|r| r.ttft_ms).sum::<f64>() / count as f64;
            let min_ttft: f64 = context_results
                .iter()
                .map(|r| r.ttft_ms)
                .fold(f64::INFINITY, f64::min);
            let max_ttft: f64 = context_results
                .iter()
                .map(|r| r.ttft_ms)
                .fold(f64::NEG_INFINITY, f64::max);
            let avg_prefill: f64 = context_results
                .iter()
                .map(|r| r.prefill_speed_tps)
                .sum::<f64>()
                / count as f64;
            let avg_decode: f64 = context_results
                .iter()
                .map(|r| r.decode_speed_tps)
                .sum::<f64>()
                / count as f64;
            let avg_total: f64 =
                context_results.iter().map(|r| r.total_time_ms).sum::<f64>() / count as f64;

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

/// Batch throughput benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchBenchmarkConfig {
    /// Context lengths to test (in tokens)
    pub context_lengths: Vec<usize>,
    /// Output length (in tokens)
    pub output_length: usize,
    /// Batch sizes to test (number of concurrent requests)
    pub batch_sizes: Vec<usize>,
    /// Number of iterations per test
    pub iterations: usize,
    /// Warmup iterations (not counted in results)
    pub warmup_iterations: usize,
}

impl Default for BatchBenchmarkConfig {
    fn default() -> Self {
        Self {
            context_lengths: vec![512, 2048, 4096],
            output_length: 128,
            batch_sizes: vec![1, 2, 4, 8, 16],
            iterations: 3,
            warmup_iterations: 1,
        }
    }
}

/// Result for a single batch throughput measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchBenchmarkResult {
    /// Context length (in tokens)
    pub context_length: usize,
    /// Output length (in tokens)
    pub output_length: usize,
    /// Batch size (number of concurrent requests)
    pub batch_size: usize,
    /// Iteration number (1-indexed)
    pub iteration: usize,
    /// Total time for all batch requests to complete (milliseconds)
    pub total_time_ms: f64,
    /// Total tokens generated (all requests combined)
    pub total_tokens: usize,
    /// Throughput in tokens per second (total_tokens / total_time)
    pub throughput_tps: f64,
    /// Average per-request latency (milliseconds)
    pub avg_request_latency_ms: f64,
    /// Min per-request latency (milliseconds)
    pub min_request_latency_ms: f64,
    /// Max per-request latency (milliseconds)
    pub max_request_latency_ms: f64,
    /// Average output tokens per request
    pub avg_output_tokens: usize,
}

/// Aggregated batch throughput summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchBenchmarkSummary {
    /// Context length (in tokens)
    pub context_length: usize,
    /// Output length (in tokens)
    pub output_length: usize,
    /// Batch size
    pub batch_size: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Average throughput (tokens/second)
    pub avg_throughput_tps: f64,
    /// Max throughput (tokens/second)
    pub max_throughput_tps: f64,
    /// Average per-request latency (milliseconds)
    pub avg_request_latency_ms: f64,
    /// P95 per-request latency (milliseconds)
    pub p95_request_latency_ms: f64,
}

/// Full batch benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchBenchmarkReport {
    /// Engine info
    pub engine_name: String,
    pub model_path: String,
    /// Test configuration
    pub config: BatchBenchmarkConfig,
    /// Individual results
    pub results: Vec<BatchBenchmarkResult>,
    /// Summaries by context length and batch size
    pub summaries: Vec<BatchBenchmarkSummary>,
    /// Timestamp
    pub timestamp: i64,
}

/// Batch benchmark runner
pub struct BatchBenchmarkRunner {
    engine: Arc<TurboMindCEngine>,
    config: BatchBenchmarkConfig,
}

impl BatchBenchmarkRunner {
    /// Create a new batch benchmark runner
    pub fn new(engine: Arc<TurboMindCEngine>, config: BatchBenchmarkConfig) -> Self {
        Self { engine, config }
    }

    /// Run all batch throughput benchmarks
    pub async fn run(&self) -> Result<BatchBenchmarkReport, String> {
        let mut all_results = Vec::new();

        for &context_length in &self.config.context_lengths {
            for &batch_size in &self.config.batch_sizes {
                // Warmup
                for _ in 0..self.config.warmup_iterations {
                    let _ = self
                        .run_batch_benchmark(context_length, batch_size, self.config.output_length, 0)
                        .await;
                }

                // Measured runs
                for iter in 1..=self.config.iterations {
                    let result = self
                        .run_batch_benchmark(context_length, batch_size, self.config.output_length, iter)
                        .await?;
                    all_results.push(result);
                }
            }
        }

        let summaries = self.generate_batch_summaries(&all_results);

        Ok(BatchBenchmarkReport {
            engine_name: "LMDeploy TurboMind C++ (batch)".to_string(),
            model_path: self.engine.model_path.clone(),
            config: self.config.clone(),
            results: all_results,
            summaries,
            timestamp: unix_timestamp(),
        })
    }

    /// Run a single batch throughput benchmark
    async fn run_batch_benchmark(
        &self,
        context_length: usize,
        batch_size: usize,
        output_length: usize,
        iteration: usize,
    ) -> Result<BatchBenchmarkResult, String> {
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

        // Build batch items
        let items: Vec<BatchItem> = (0..batch_size)
            .map(|i| BatchItem {
                request_id: i as u64,
                prompt: prompt.clone(),
                params: GenerationParams {
                    max_tokens: Some(output_length),
                    ..Default::default()
                },
                need_logprobs: false,
            })
            .collect();

        let start = Instant::now();
        let results = self.engine.generate_batch(items).await;
        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let total_tokens: usize = results.iter().map(|r| r.num_tokens).sum();
        let throughputs: Vec<f64> = results
            .iter()
            .filter(|r| r.elapsed_ms > 0.0)
            .map(|r| (r.num_tokens as f64 * 1000.0) / r.elapsed_ms)
            .collect();

        let avg_throughput = if !throughputs.is_empty() {
            throughputs.iter().sum::<f64>() / throughputs.len() as f64
        } else {
            0.0
        };

        let request_latencies: Vec<f64> = results.iter().map(|r| r.elapsed_ms).collect();
        let avg_latency = if !request_latencies.is_empty() {
            request_latencies.iter().sum::<f64>() / request_latencies.len() as f64
        } else {
            0.0
        };
        let min_latency = request_latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_latency = request_latencies
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let avg_output = if !results.is_empty() {
            total_tokens / results.len()
        } else {
            0
        };

        Ok(BatchBenchmarkResult {
            context_length: actual_input_tokens,
            output_length,
            batch_size,
            iteration,
            total_time_ms,
            total_tokens,
            throughput_tps: avg_throughput,
            avg_request_latency_ms: avg_latency,
            min_request_latency_ms: min_latency,
            max_request_latency_ms: max_latency,
            avg_output_tokens: avg_output,
        })
    }

    /// Generate summaries from batch results
    fn generate_batch_summaries(
        &self,
        results: &[BatchBenchmarkResult],
    ) -> Vec<BatchBenchmarkSummary> {
        let mut summaries = Vec::new();

        for &target_context in &self.config.context_lengths {
            let tolerance = match target_context {
                0..=4096 => 200,
                4097..=16384 => 1000,
                _ => 3000,
            };

            for &batch_size in &self.config.batch_sizes {
                let filtered: Vec<_> = results
                    .iter()
                    .filter(|r| {
                        r.batch_size == batch_size
                            && (r.context_length as isize - target_context as isize).abs() < tolerance
                    })
                    .collect();

                if filtered.is_empty() {
                    continue;
                }

                let count = filtered.len();
                let avg_throughput =
                    filtered.iter().map(|r| r.throughput_tps).sum::<f64>() / count as f64;
                let max_throughput = filtered
                    .iter()
                    .map(|r| r.throughput_tps)
                    .fold(f64::NEG_INFINITY, f64::max);
                let avg_latency =
                    filtered.iter().map(|r| r.avg_request_latency_ms).sum::<f64>() / count as f64;

                let mut sorted_latencies: Vec<f64> =
                    filtered.iter().map(|r| r.avg_request_latency_ms).collect();
                sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p95_latency = *sorted_latencies
                    .get((sorted_latencies.len() as f64 * 0.95) as usize)
                    .unwrap_or(&0.0);

                summaries.push(BatchBenchmarkSummary {
                    context_length: target_context,
                    output_length: self.config.output_length,
                    batch_size,
                    iterations: count,
                    avg_throughput_tps: avg_throughput,
                    max_throughput_tps: max_throughput,
                    avg_request_latency_ms: avg_latency,
                    p95_request_latency_ms: p95_latency,
                });
            }
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
        assert_eq!(
            config.context_lengths,
            vec![1024, 4096, 8192, 16384, 32768, 49152, 65536, 131072]
        );
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
            actual_output_tokens: 512,
            itl_ms: vec![1.5, 1.2, 1.3],
        };

        let json = serde_json::to_string(&result).unwrap();
        let _parsed: BenchmarkResult = serde_json::from_str(&json).unwrap();
    }
}
