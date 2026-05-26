// Rust Server prefill benchmark using GPU tokenizer zero-copy path
// This benchmark uses the existing DLPack zero-copy path that should be faster
// than the default Vec<u32> → Pinned → GPU path.

use std::env;
use std::time::Instant;

use lmdeploy_rust_server::tokenizer::LMTokenizerWithGpu;
use lmdeploy_rust_server::model::cpp_engine::{
    TurboMindCEngine, GenerationParams, DlpackInputTensor, DlpackDtype, DlpackDevice,
};
use lmdeploy_rust_server::turbomind_c::TurboMind;

const MODEL: &str = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ";
const REPEAT_TEXT: &str = "The quick brown fox jumps over the lazy dog. ";

fn gen_prompt(token_count: usize) -> String {
    let repeats = (1).max(token_count / 10);
    REPEAT_TEXT.repeat(repeats)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Rust Server GPU Tokenizer Prefill Benchmark");
    println!("Model: {}", MODEL);

    let test_cases = vec![
        ("1K", 1024),
        ("4K", 4096),
        ("8K", 8192),
    ];

    // Create engine
    let engine = TurboMindCEngine::new(MODEL, 16384, 1, 0.4).await?;
    let gpu_tokenizer = LMTokenizerWithGpu::from_path(MODEL)?;

    println!("\n{:>8} | {:>10} | {:>12} | {:>12} | {:>15}",
             "Context", "Tokens", "TTFT (ms)", "Min (ms)", "Prefill (tok/s)");
    println!("{}", "-".repeat(70));

    for (label, target_tokens) in &test_cases {
        let prompt = gen_prompt(*target_tokens);

        // Warmup
        for _ in 0..2 {
            let warmup_prompt = gen_prompt(512);
            let gpu_tensor = gpu_tokenizer.encode_to_gpu(&warmup_prompt, false)?;
            if let Some(ref event) = gpu_tensor.sync_event {
                let _ = event.sync();
            }
            let dlpack_input = DlpackInputTensor {
                name: "input_ids",
                data: gpu_tensor.gpu_ptr as *const std::ffi::c_void,
                shape: vec![gpu_tensor.len as i64],
                dtype: DlpackDtype::UInt(32),
                device: DlpackDevice::Cuda(0),
            };
            let _ = engine.generate_with_dlpack_input(dlpack_input, GenerationParams {
                max_tokens: Some(10),
                temperature: Some(0.7),
                ..Default::default()
            }).await;
        }

        let mut times = Vec::new();
        let actual_tokens = gpu_tokenizer.encode(&prompt, false, false)?.len();

        for r in 0..3 {
            let gpu_tensor = gpu_tokenizer.encode_to_gpu(&prompt, false)?;

            let start = Instant::now();

            // Async copy happens in parallel, but we must sync before use
            if let Some(ref event) = gpu_tensor.sync_event {
                let _ = event.sync();
            }

            let dlpack_input = DlpackInputTensor {
                name: "input_ids",
                data: gpu_tensor.gpu_ptr as *const std::ffi::c_void,
                shape: vec![gpu_tensor.len as i64],
                dtype: DlpackDtype::UInt(32),
                device: DlpackDevice::Cuda(0),
            };

            let (text, _, elapsed_ms) = engine.generate_with_dlpack_input_and_metrics(dlpack_input, GenerationParams {
                max_tokens: Some(10),
                temperature: Some(0.7),
                ..Default::default()
            }).await;

            let _ = text;
            times.push(elapsed_ms);
            println!("{} Run {}: {:.1}ms -> {:.0} tok/s",
                     label, r, elapsed_ms, actual_tokens as f64 / elapsed_ms * 1000.0);
        }

        if !times.is_empty() {
            let avg_ms = times.iter().sum::<f64>() / times.len() as f64;
            let min_ms = times.iter().fold(times[0], |a, b| a.min(*b));
            let tps = actual_tokens as f64 / avg_ms * 1000.0;
            println!("{} AVG: {:.1}ms ({:.0} tok/s, min {:.1}ms)", label, avg_ms, tps, min_ms);
        }
    }

    Ok(())
}
