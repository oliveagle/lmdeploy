//! Python Bridge for TurboMind inference
//!
//! This module provides a bridge to the Python TurboMind API via a subprocess.
//! The Python subprocess loads the model using the working Python API and
//! exposes inference via stdin/stdout JSON protocol.

use futures::{Stream, StreamExt};
use std::io::{BufRead, BufReader, Write};
use std::pin::Pin;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Python bridge protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd")]
enum BridgeCommand {
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "load")]
    Load {
        model_path: String,
        #[serde(default)]
        engine_config: EngineConfig,
    },
    #[serde(rename = "generate")]
    Generate {
        input_ids: Vec<u32>,
        #[serde(default = "default_max_new_tokens")]
        max_new_tokens: usize,
        #[serde(default = "default_temperature")]
        temperature: f32,
        #[serde(default = "default_top_p")]
        top_p: f32,
        #[serde(default = "default_top_k")]
        top_k: i32,
    },
    #[serde(rename = "generate_stream")]
    GenerateStream {
        input_ids: Vec<u32>,
        #[serde(default = "default_max_new_tokens")]
        max_new_tokens: usize,
        #[serde(default = "default_temperature")]
        temperature: f32,
        #[serde(default = "default_top_p")]
        top_p: f32,
        #[serde(default = "default_top_k")]
        top_k: i32,
    },
    #[serde(rename = "metrics")]
    Metrics,
    #[serde(rename = "shutdown")]
    Shutdown,
}

fn default_max_new_tokens() -> usize {
    100
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.95
}

fn default_top_k() -> i32 {
    50
}

/// Bridge response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BridgeResponse {
    status: String,
    #[serde(default)]
    output_ids: Vec<u32>,
    #[serde(default)]
    elapsed_ms: f32,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    metrics: Option<ScheduleMetrics>,
}

/// Streaming response chunk from Python bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamChunkResponse {
    status: String,
    #[serde(default)]
    token_id: u32,
    #[serde(default)]
    text: String,
    #[serde(default)]
    is_first: bool,
    #[serde(default)]
    is_last: bool,
    #[serde(default)]
    ttft_ms: Option<f64>,
    #[serde(default)]
    elapsed_ms: Option<f64>,
    #[serde(default)]
    token_index: Option<usize>,
    #[serde(default)]
    r#type: Option<String>, // "chunk" or "done"
    #[serde(default)]
    error: Option<String>,
}

/// A single chunk of streaming output from the bridge
#[derive(Debug, Clone)]
pub struct BridgeStreamChunk {
    pub token_id: u32,
    pub text: String,
    pub is_last: bool,
    pub ttft_ms: Option<f64>,
}

impl From<StreamChunkResponse> for BridgeStreamChunk {
    fn from(chunk: StreamChunkResponse) -> Self {
        let is_done = chunk.r#type.as_deref() == Some("done") || chunk.is_last;
        Self {
            token_id: chunk.token_id,
            text: chunk.text,
            is_last: is_done,
            ttft_ms: chunk.ttft_ms,
        }
    }
}

/// Schedule metrics from TurboMind
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScheduleMetrics {
    pub total_seqs: i32,
    pub active_seqs: i32,
    pub waiting_seqs: i32,
    pub total_blocks: i32,
    pub active_blocks: i32,
    pub cached_blocks: i32,
    pub free_blocks: i32,
}

/// Engine config for bridge
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct EngineConfig {
    #[serde(default)]
    session_len: i32,
    #[serde(default)]
    tp: i32,
    #[serde(default)]
    quant_policy: i32,
    #[serde(default)]
    max_batch_size: i32,
    #[serde(default)]
    cache_block_seq_len: i32,
}

/// Python bridge subprocess wrapper
pub struct PythonBridge {
    child: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
    model_path: String,
}

impl PythonBridge {
    /// Get the model path this bridge is serving
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Create a new Python bridge subprocess
    pub fn new(model_path: &str, session_len: i32, tp: i32, quant_policy: i32) -> Result<Self> {
        tracing::info!(
            model_path = %model_path,
            session_len,
            tp,
            quant_policy,
            "Starting Python bridge subprocess"
        );

        // Find the bridge script using canonical path resolution
        // Use CARGO_MANIFEST_DIR to locate the lmdeploy package directory reliably
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap());

        // First try: <manifest_dir>/../lmdeploy/turbomind/python_bridge.py
        // This works when running from lmdeploy-rust-server/ directory
        let bridge_script = manifest_dir
            .join("../lmdeploy/turbomind/python_bridge.py")
            .canonicalize()
            .ok();

        // Second try: <manifest_dir>/lmdeploy/turbomind/python_bridge.py
        // This works when running from the lmdeploy root directory
        let bridge_script = bridge_script.or_else(|| {
            manifest_dir
                .join("lmdeploy/turbomind/python_bridge.py")
                .canonicalize()
                .ok()
        });

        // Third try: environment variable LMDEPLOY_BRIDGE_SCRIPT
        let bridge_script = bridge_script.or_else(|| {
            std::env::var("LMDEPLOY_BRIDGE_SCRIPT")
                .map(std::path::PathBuf::from)
                .ok()
                .filter(|p| p.exists())
        });

        let bridge_script = match bridge_script {
            Some(path) => path,
            None => {
                // Provide helpful error message with debug info
                let debug_paths = vec![
                    manifest_dir.join("../lmdeploy/turbomind/python_bridge.py"),
                    manifest_dir.join("lmdeploy/turbomind/python_bridge.py"),
                ];
                return Err(crate::error::AppError::ModelLoadFailed(format!(
                    "Python bridge script not found. Searched:\n  - {}\n  - {}\n\n\
                    Set LMDEPLOY_BRIDGE_SCRIPT env var to specify the exact path.",
                    debug_paths[0].display(),
                    debug_paths[1].display()
                )));
            }
        };

        tracing::debug!(bridge_script = %bridge_script.display(), "Resolved Python bridge script path");

        // Setup PYTHONPATH for lmdeploy imports
        let pythonpath = std::env::var("PYTHONPATH").unwrap_or_default();
        let mut cmd = Command::new("python3");

        // Add lmdeploy directory to PYTHONPATH
        let lmdeploy_path = bridge_script
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/mnt/data/lmdeploy".to_string());

        cmd.env("PYTHONPATH", format!("{}:{}", lmdeploy_path, pythonpath));

        // Start the Python subprocess
        let mut child = cmd
            .arg(&bridge_script)
            .arg("--model-path")
            .arg(model_path)
            .arg("--session-len")
            .arg(session_len.to_string())
            .arg("--tp")
            .arg(tp.to_string())
            .arg("--quant-policy")
            .arg(quant_policy.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                crate::error::AppError::ModelLoadFailed(format!(
                    "Failed to start Python bridge: {}",
                    e
                ))
            })?;

        let stdin = child.stdin.take().expect("Failed to get stdin");
        let stdout = child.stdout.take().expect("Failed to get stdout");
        let stdout = BufReader::new(stdout);

        let bridge = Self {
            child: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(stdout)),
            model_path: model_path.to_string(),
        };

        // Wait for the model to load
        tracing::info!("Waiting for Python bridge to load model...");
        let response = bridge.read_response()?;
        if response.status != "ok" {
            return Err(crate::error::AppError::ModelLoadFailed(format!(
                "Failed to load model in Python bridge: {:?}",
                response.message
            )));
        }

        tracing::info!("Python bridge ready");
        Ok(bridge)
    }

    /// Send a command and read the response
    fn send_command(&self, cmd: &BridgeCommand) -> Result<BridgeResponse> {
        let json = serde_json::to_string(cmd).map_err(|e| {
            crate::error::AppError::InferenceFailed(format!("JSON encode error: {}", e))
        })?;

        let mut stdin = self
            .stdin
            .lock()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Lock error: {}", e)))?;
        writeln!(stdin, "{}", json)
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Write error: {}", e)))?;
        stdin
            .flush()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Flush error: {}", e)))?;
        drop(stdin);

        self.read_response()
    }

    /// Read a response from the bridge
    fn read_response(&self) -> Result<BridgeResponse> {
        let mut stdout = self
            .stdout
            .lock()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Lock error: {}", e)))?;
        let mut line = String::new();
        stdout
            .read_line(&mut line)
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Read error: {}", e)))?;
        drop(stdout);

        let response: BridgeResponse = serde_json::from_str(&line).map_err(|e| {
            crate::error::AppError::InferenceFailed(format!("JSON decode error: {}", e))
        })?;

        if response.status != "ok" {
            return Err(crate::error::AppError::InferenceFailed(
                response
                    .message
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        Ok(response)
    }

    /// Generate tokens from input_ids, returning (output_ids, elapsed_ms)
    pub fn generate_with_metrics(
        &self,
        input_ids: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<(Vec<u32>, f64)> {
        let cmd = BridgeCommand::Generate {
            input_ids,
            max_new_tokens,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
        };

        let response = self.send_command(&cmd)?;
        Ok((response.output_ids, response.elapsed_ms as f64))
    }

    /// Generate tokens from input_ids
    pub fn generate(&self, input_ids: Vec<u32>, max_new_tokens: usize) -> Result<Vec<u32>> {
        let (ids, _) = self.generate_with_metrics(input_ids, max_new_tokens)?;
        Ok(ids)
    }

    /// Get schedule metrics
    pub fn get_metrics(&self) -> Result<ScheduleMetrics> {
        let response = self.send_command(&BridgeCommand::Metrics)?;
        Ok(response.metrics.unwrap_or_default())
    }

    /// Generate tokens from input_ids with streaming output
    /// Returns a stream of (token_id, text, is_last, ttft_ms) chunks
    pub fn generate_stream(
        &self,
        input_ids: Vec<u32>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: i32,
    ) -> Result<Pin<Box<dyn Stream<Item = BridgeStreamChunk> + Send>>> {
        // Send the generate_stream command
        let cmd = BridgeCommand::GenerateStream {
            input_ids,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        };

        let json = serde_json::to_string(&cmd).map_err(|e| {
            crate::error::AppError::InferenceFailed(format!("JSON encode error: {}", e))
        })?;

        let mut stdin = self
            .stdin
            .lock()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Lock error: {}", e)))?;
        writeln!(stdin, "{}", json)
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Write error: {}", e)))?;
        stdin
            .flush()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Flush error: {}", e)))?;
        drop(stdin);

        // Spawn a task to read streaming responses
        let (tx, rx) = mpsc::channel::<BridgeStreamChunk>(1024);
        let stdout_clone = self.stdout.clone();
        std::thread::spawn(move || {
            let mut stdout = stdout_clone.lock().unwrap();
            let mut line = String::new();
            loop {
                line.clear();
                match stdout.read_line(&mut line) {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        let line_trimmed = line.trim();
                        if line_trimmed.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<StreamChunkResponse>(line_trimmed) {
                            Ok(chunk) => {
                                let is_done =
                                    chunk.r#type.as_deref() == Some("done") || chunk.is_last;
                                let stream_chunk = BridgeStreamChunk {
                                    token_id: chunk.token_id,
                                    text: chunk.text.clone(),
                                    is_last: is_done,
                                    ttft_ms: chunk.ttft_ms,
                                };

                                // Send the chunk; if receiver dropped, stop
                                if tx.blocking_send(stream_chunk).is_err() {
                                    tracing::debug!("Stream receiver dropped, stopping");
                                    break;
                                }

                                if is_done {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, line = ?line_trimmed, "Failed to parse stream chunk");
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Error reading stream output");
                        break;
                    }
                }
            }
        });

        let stream = ReceiverStream::new(rx).map(|chunk| BridgeStreamChunk {
            token_id: chunk.token_id,
            text: chunk.text,
            is_last: chunk.is_last,
            ttft_ms: chunk.ttft_ms,
        });

        Ok(Box::pin(stream))
    }

    /// Shutdown the bridge
    pub fn shutdown(&self) -> Result<()> {
        let _ = self.send_command(&BridgeCommand::Shutdown);

        let mut child = self
            .child
            .lock()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Lock error: {}", e)))?;
        let _ = child.kill();
        let _ = child.wait();
        Ok(())
    }
}

impl Drop for PythonBridge {
    fn drop(&mut self) {
        // Try to shutdown gracefully
        let _ = self.send_command(&BridgeCommand::Shutdown);
        let mut child = self.child.lock().unwrap();
        let _ = child.kill();
        let _ = child.wait();
    }
}
