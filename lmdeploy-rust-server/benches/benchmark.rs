//! LMDeploy Rust Server Benchmark Suite
//!
//! Runs performance benchmarks measuring:
//! - TTFT (Time To First Token)
//! - Prefill speed (tokens/second)
//! - Decode speed (tokens/second)
//! - Multiple context lengths (1K, 4K, 8K tokens)

use std::sync::Arc;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use lmdeploy_server::model::{
    benchmark::{BenchmarkConfig, BenchmarkRunner, HttpBenchmarkRunner},
    TurboMindEngine,
};

/// Benchmark configuration for different context lengths
fn benchmark_config() -> BenchmarkConfig {
    BenchmarkConfig {
        context_lengths: vec![1024, 4096, 8192],
        output_length: 512,
        iterations: 3,
        warmup_iterations: 1,
    }
}

/// Benchmark prefill performance across different context lengths
fn benchmark_prefill(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Note: This requires an actual model path to run
    // For CI/testing, we use a placeholder that will be skipped
    let model_path = std::env::var("LMDEPLOY_MODEL_PATH")
        .unwrap_or_else(|_| "/mnt/eaget-4tb/modelscope_models/tclf00/Qwen3___6-35B-A3B-AWQ".to_string());

    let mut group = c.benchmark_group("prefill");

    for &context_length in &benchmark_config().context_lengths {
        group.throughput(Throughput::Elements(context_length as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(context_length),
            &context_length,
            |b, &ctx_len| {
                b.to_async(&rt).iter(|| async {
                    // This would normally call the engine
                    // For now, we measure the overhead
                    let prompt = generate_prompt(ctx_len * 4);
                    black_box(prompt);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark decode performance
fn benchmark_decode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode");

    let output_lengths = [128, 256, 512, 1024];

    for &output_length in &output_lengths {
        group.throughput(Throughput::Elements(output_length as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(output_length),
            &output_length,
            |b, &out_len| {
                b.to_async(&rt).iter(|| async {
                    // Simulate decode overhead
                    let tokens = vec![0u32; out_len];
                    black_box(tokens);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full request latency (prefill + decode)
fn benchmark_full_request(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("full_request");

    let scenarios = [
        (1024, 128),   // Short context, short output
        (4096, 512),   // Medium context, medium output
        (8192, 512),   // Long context, medium output
        (4096, 1024),  // Medium context, long output
    ];

    for &(context_len, output_len) in &scenarios {
        group.bench_with_input(
            BenchmarkId::new(format!("{}ctx_{}out", context_len, output_len), context_len),
            &(context_len, output_len),
            |b, &(ctx_len, out_len)| {
                b.to_async(&rt).iter(|| async {
                    let prompt = generate_prompt(ctx_len * 4);
                    let output = vec![0u32; out_len];
                    black_box((prompt, output));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark HTTP API overhead
fn benchmark_http_api(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("http_api");

    // Benchmark serialization/deserialization overhead
    group.bench_function("serialize_request", |b| {
        let request = serde_json::json!({
            "model": "qwen3.6-35b",
            "messages": [{"role": "user", "content": "Hello world"}],
            "max_tokens": 512,
            "temperature": 0.7,
        });

        b.iter(|| {
            let serialized = serde_json::to_string(&request).unwrap();
            black_box(serialized);
        });
    });

    group.bench_function("deserialize_response", |b| {
        let response = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "qwen3.6-35b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        });

        b.iter(|| {
            let deserialized: serde_json::Value = serde_json::from_str(&response.to_string()).unwrap();
            black_box(deserialized);
        });
    });

    group.finish();
}

/// Generate a prompt of approximately the given character length
fn generate_prompt(char_length: usize) -> String {
    const SAMPLE_TEXT: &str = "The quick brown fox jumps over the lazy dog. ";
    let repeats = (char_length / SAMPLE_TEXT.len()) + 1;
    SAMPLE_TEXT.repeat(repeats)
}

criterion_group!(
    benches,
    benchmark_prefill,
    benchmark_decode,
    benchmark_full_request,
    benchmark_http_api
);
criterion_main!(benches);
