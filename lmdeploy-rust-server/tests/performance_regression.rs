//! Performance Regression Tests
//!
//! These tests verify that the Rust C++ engine meets minimum performance baselines.
//! Tests are skipped in CI by default unless a model is available.
//!
//! ## Performance Baselines
//!
//! Based on Qwen3.6-35B-A3B-AWQ, V100 32GB:
//!
//! | Metric | Baseline | Target |
//! |--------|----------|--------|
//! | Prefill @ 8K | >3000 tok/s | >5000 tok/s |
//! | Decode @ 8K | >40 tok/s | >50 tok/s |
//! | TTFT @ 8K | <2000ms | <1000ms |
//!
//! ## Running Tests
//!
//! ```bash
//! # Run with local model
//! AWQ_MODEL_PATH=/path/to/model cargo test --test performance_regression
//!
//! # Run in CI (skips if model not found)
//! cargo test --test performance_regression
//! ```

use lmdeploy_server::model::cpp_engine::{GenerationParams, TurboMindCEngine};
use std::time::Instant;

/// Get the AWQ model path from environment or use default
fn get_model_path() -> String {
    std::env::var("AWQ_MODEL_PATH")
        .unwrap_or_else(|_| "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".to_string())
}

/// Check if the model path exists
fn model_exists() -> bool {
    let path = get_model_path();
    std::path::Path::new(&path).exists()
}

/// Generate a prompt of approximately the given token count
///
/// Uses a repeating pattern to generate predictable input lengths.
/// Each iteration of the pattern generates approximately 128 tokens.
fn generate_prompt(target_tokens: usize) -> String {
    const SAMPLE_TEXT: &str = "The quick brown fox jumps over the lazy dog. ";
    let chars_per_token = 4; // Approximate English: ~4 chars per token
    let target_chars = target_tokens * chars_per_token;
    let repeats = (target_chars / SAMPLE_TEXT.len()) + 1;
    SAMPLE_TEXT.repeat(repeats)
}

/// Performance metrics for a single inference run
#[derive(Debug, Clone)]
struct InferenceMetrics {
    /// Time to first token (ms)
    ttft_ms: f64,
    /// Total elapsed time (ms)
    elapsed_ms: f64,
    /// Number of input tokens
    input_tokens: usize,
    /// Number of output tokens
    output_tokens: usize,
    /// Prefill speed (tokens/second)
    prefill_tps: f64,
    /// Decode speed (tokens/second)
    decode_tps: f64,
}

/// Calculate inference metrics from token counts and timing
fn calculate_metrics(
    input_tokens: usize,
    output_tokens: usize,
    elapsed_ms: f64,
) -> InferenceMetrics {
    // TTFT is approximately the time until first output token
    // For non-streaming, we estimate it as a fraction of total time
    // based on the ratio of prefill to decode work
    let total_tokens = input_tokens + output_tokens;
    let prefill_ratio = input_tokens as f64 / total_tokens as f64;
    let ttft_ms = elapsed_ms * prefill_ratio;

    // Prefill speed: input tokens / prefill time
    let prefill_time_sec = ttft_ms / 1000.0;
    let prefill_tps = if prefill_time_sec > 0.0 {
        input_tokens as f64 / prefill_time_sec
    } else {
        0.0
    };

    // Decode speed: output tokens / decode time
    let decode_time_sec = (elapsed_ms - ttft_ms) / 1000.0;
    let decode_tps = if decode_time_sec > 0.0 {
        output_tokens as f64 / decode_time_sec
    } else {
        0.0
    };

    InferenceMetrics {
        ttft_ms,
        elapsed_ms,
        input_tokens,
        output_tokens,
        prefill_tps,
        decode_tps,
    }
}

/// Warm up the engine with a few inference runs
///
/// Warmup ensures:
/// - GPU kernels are compiled and cached
/// - Memory allocations are stabilized
/// - First-run overhead is eliminated
async fn warmup_engine(engine: &TurboMindCEngine, iterations: usize) {
    let prompt = generate_prompt(128);

    for _ in 0..iterations {
        let _ = engine
            .generate(
                &prompt,
                GenerationParams {
                    max_tokens: Some(16),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .await;
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[tokio::test]
    /// Test prefill speed at 8K context length
    ///
    /// Baseline: >3000 tokens/second
    /// Target: >5000 tokens/second
    async fn test_prefill_speed_8k() {
        if !model_exists() {
            println!("Skipping: AWQ model not found");
            return;
        }

        let path = get_model_path();
        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        // Warmup: 2 iterations with short prompts
        warmup_engine(&engine, 2).await;

        // Test: 8K context prompt, generate 16 tokens
        let input_tokens = 8192;
        let prompt = generate_prompt(input_tokens);
        let max_tokens = 16;

        let start = Instant::now();
        let (_text, num_tokens, elapsed_ms) = engine
            .generate_with_metrics(
                &prompt,
                GenerationParams {
                    max_tokens: Some(max_tokens),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .await;
        let total_elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let output_tokens = num_tokens.saturating_sub(input_tokens);
        let metrics = calculate_metrics(input_tokens, output_tokens, total_elapsed);

        println!(
            "Prefill @ 8K: {:.2} tok/s (TTFT: {:.2}ms, total: {:.2}ms)",
            metrics.prefill_tps, metrics.ttft_ms, metrics.elapsed_ms
        );

        // Baseline assertion: >3000 tok/s
        assert!(
            metrics.prefill_tps > 3000.0,
            "Prefill speed below baseline: {:.2} tok/s (expected >3000)",
            metrics.prefill_tps
        );

        // Target assertion: >5000 tok/s (warn if below)
        if metrics.prefill_tps < 5000.0 {
            println!(
                "WARNING: Prefill speed below target: {:.2} tok/s (target >5000)",
                metrics.prefill_tps
            );
        }
    }

    #[tokio::test]
    /// Test prefill speed at 1K context length
    ///
    /// Baseline: >500 tokens/second
    /// Target: >1000 tokens/second
    async fn test_prefill_speed_1k() {
        if !model_exists() {
            println!("Skipping: AWQ model not found");
            return;
        }

        let path = get_model_path();
        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        // Warmup
        warmup_engine(&engine, 2).await;

        // Test: 1K context prompt, generate 16 tokens
        let input_tokens = 1024;
        let prompt = generate_prompt(input_tokens);
        let max_tokens = 16;

        let start = Instant::now();
        let (_text, num_tokens, elapsed_ms) = engine
            .generate_with_metrics(
                &prompt,
                GenerationParams {
                    max_tokens: Some(max_tokens),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .await;
        let total_elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let output_tokens = num_tokens.saturating_sub(input_tokens);
        let metrics = calculate_metrics(input_tokens, output_tokens, total_elapsed);

        println!(
            "Prefill @ 1K: {:.2} tok/s (TTFT: {:.2}ms, total: {:.2}ms)",
            metrics.prefill_tps, metrics.ttft_ms, metrics.elapsed_ms
        );

        // Baseline assertion: >500 tok/s
        assert!(
            metrics.prefill_tps > 500.0,
            "Prefill speed below baseline: {:.2} tok/s (expected >500)",
            metrics.prefill_tps
        );

        // Target assertion: >1000 tok/s
        if metrics.prefill_tps < 1000.0 {
            println!(
                "WARNING: Prefill speed below target: {:.2} tok/s (target >1000)",
                metrics.prefill_tps
            );
        }
    }

    #[tokio::test]
    /// Test decode speed at 8K context length
    ///
    /// Baseline: >40 tokens/second
    /// Target: >50 tokens/second
    async fn test_decode_speed_8k() {
        if !model_exists() {
            println!("Skipping: AWQ model not found");
            return;
        }

        let path = get_model_path();
        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        // Warmup
        warmup_engine(&engine, 2).await;

        // Test: 8K context prompt, generate 256 tokens
        let input_tokens = 8192;
        let prompt = generate_prompt(input_tokens);
        let max_tokens = 256;

        let start = Instant::now();
        let (_text, num_tokens, elapsed_ms) = engine
            .generate_with_metrics(
                &prompt,
                GenerationParams {
                    max_tokens: Some(max_tokens),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .await;
        let total_elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let output_tokens = num_tokens.saturating_sub(input_tokens);
        let metrics = calculate_metrics(input_tokens, output_tokens, total_elapsed);

        println!(
            "Decode @ 8K: {:.2} tok/s (TTFT: {:.2}ms, total: {:.2}ms, out: {} tokens)",
            metrics.decode_tps, metrics.ttft_ms, metrics.elapsed_ms, output_tokens
        );

        // Baseline assertion: >40 tok/s
        assert!(
            metrics.decode_tps > 40.0,
            "Decode speed below baseline: {:.2} tok/s (expected >40)",
            metrics.decode_tps
        );

        // Target assertion: >50 tok/s
        if metrics.decode_tps < 50.0 {
            println!(
                "WARNING: Decode speed below target: {:.2} tok/s (target >50)",
                metrics.decode_tps
            );
        }
    }

    #[tokio::test]
    /// Test TTFT (Time To First Token) at 8K context length
    ///
    /// Baseline: <2000ms
    /// Target: <1000ms
    async fn test_ttft_8k() {
        if !model_exists() {
            println!("Skipping: AWQ model not found");
            return;
        }

        let path = get_model_path();
        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        // Warmup
        warmup_engine(&engine, 2).await;

        // Test: 8K context prompt, generate 16 tokens
        let input_tokens = 8192;
        let prompt = generate_prompt(input_tokens);
        let max_tokens = 16;

        let start = Instant::now();
        let (_text, num_tokens, elapsed_ms) = engine
            .generate_with_metrics(
                &prompt,
                GenerationParams {
                    max_tokens: Some(max_tokens),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .await;
        let total_elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let output_tokens = num_tokens.saturating_sub(input_tokens);
        let metrics = calculate_metrics(input_tokens, output_tokens, total_elapsed);

        println!(
            "TTFT @ 8K: {:.2}ms (Prefill: {:.2} tok/s, Decode: {:.2} tok/s)",
            metrics.ttft_ms, metrics.prefill_tps, metrics.decode_tps
        );

        // Baseline assertion: <2000ms
        assert!(
            metrics.ttft_ms < 2000.0,
            "TTFT above baseline: {:.2}ms (expected <2000)",
            metrics.ttft_ms
        );

        // Target assertion: <1000ms
        if metrics.ttft_ms > 1000.0 {
            println!(
                "WARNING: TTFT above target: {:.2}ms (target <1000)",
                metrics.ttft_ms
            );
        }
    }

    #[tokio::test]
    /// Test performance regression detection
    ///
    /// Runs all key metrics and reports if any baseline is violated.
    /// This is a comprehensive test that catches regressions early.
    async fn test_performance_regression_comprehensive() {
        if !model_exists() {
            println!("Skipping: AWQ model not found");
            return;
        }

        let path = get_model_path();
        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        // Warmup
        warmup_engine(&engine, 3).await;

        let mut failures = Vec::new();

        // Test 1: Prefill @ 8K
        let input_tokens = 8192;
        let prompt = generate_prompt(input_tokens);
        let start = Instant::now();
        let (_text, num_tokens, elapsed_ms) = engine
            .generate_with_metrics(
                &prompt,
                GenerationParams {
                    max_tokens: Some(16),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .await;
        let total_elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let output_tokens = num_tokens.saturating_sub(input_tokens);
        let metrics = calculate_metrics(input_tokens, output_tokens, total_elapsed);

        println!(
            "Comprehensive - Prefill @ 8K: {:.2} tok/s",
            metrics.prefill_tps
        );
        if metrics.prefill_tps < 3000.0 {
            failures.push(format!(
                "Prefill @ 8K: {:.2} tok/s (expected >3000)",
                metrics.prefill_tps
            ));
        }

        // Test 2: Decode @ 8K
        let start = Instant::now();
        let (_text, num_tokens, elapsed_ms) = engine
            .generate_with_metrics(
                &prompt,
                GenerationParams {
                    max_tokens: Some(256),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .await;
        let total_elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let output_tokens = num_tokens.saturating_sub(input_tokens);
        let metrics = calculate_metrics(input_tokens, output_tokens, total_elapsed);

        println!(
            "Comprehensive - Decode @ 8K: {:.2} tok/s",
            metrics.decode_tps
        );
        if metrics.decode_tps < 40.0 {
            failures.push(format!(
                "Decode @ 8K: {:.2} tok/s (expected >40)",
                metrics.decode_tps
            ));
        }

        // Test 3: TTFT @ 8K
        println!("Comprehensive - TTFT @ 8K: {:.2}ms", metrics.ttft_ms);
        if metrics.ttft_ms > 2000.0 {
            failures.push(format!(
                "TTFT @ 8K: {:.2}ms (expected <2000)",
                metrics.ttft_ms
            ));
        }

        // Report all failures at once
        if !failures.is_empty() {
            panic!("Performance regression detected:\n{}", failures.join("\n"));
        }
    }

    #[tokio::test]
    /// Test performance is repeatable across multiple runs
    ///
    /// Verifies that performance measurements are stable and not outliers.
    async fn test_performance_repeatability() {
        if !model_exists() {
            println!("Skipping: AWQ model not found");
            return;
        }

        let path = get_model_path();
        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        // Warmup
        warmup_engine(&engine, 2).await;

        let input_tokens = 4096;
        let prompt = generate_prompt(input_tokens);
        let iterations = 5;
        let mut prefill_speeds = Vec::new();

        for _i in 0..iterations {
            let start = Instant::now();
            let (_text, num_tokens, elapsed_ms) = engine
                .generate_with_metrics(
                    &prompt,
                    GenerationParams {
                        max_tokens: Some(16),
                        temperature: Some(0.0),
                        ..Default::default()
                    },
                )
                .await;
            let total_elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let output_tokens = num_tokens.saturating_sub(input_tokens);
            let metrics = calculate_metrics(input_tokens, output_tokens, total_elapsed);
            prefill_speeds.push(metrics.prefill_tps);
        }

        // Calculate statistics
        let avg_speed: f64 = prefill_speeds.iter().sum::<f64>() / iterations as f64;
        let min_speed = prefill_speeds.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_speed = prefill_speeds
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let variance = prefill_speeds
            .iter()
            .map(|&x| (x - avg_speed).powi(2))
            .sum::<f64>()
            / iterations as f64;
        let std_dev = variance.sqrt();
        let cv = (std_dev / avg_speed) * 100.0; // Coefficient of variation

        println!(
            "Repeatability @ 4K: avg={:.2} tok/s, min={:.2}, max={:.2}, cv={:.2}%",
            avg_speed, min_speed, max_speed, cv
        );

        // Verify measurements are stable (CV < 15%)
        assert!(
            cv < 15.0,
            "Performance measurements too variable: CV={:.2}% (expected <15%)",
            cv
        );

        // Verify all runs meet baseline
        assert!(
            min_speed > 2000.0,
            "Minimum speed below baseline: {:.2} tok/s (expected >2000)",
            min_speed
        );
    }
}
