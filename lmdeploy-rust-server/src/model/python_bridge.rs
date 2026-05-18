//! Python Bridge for TurboMind inference
//!
//! This module provides a bridge to the Python TurboMind API via a subprocess.
//! The Python subprocess loads the model using the working Python API and
//! exposes inference via stdin/stdout JSON protocol.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

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
    /// Create a new Python bridge subprocess
    pub fn new(model_path: &str, session_len: i32, tp: i32, quant_policy: i32) -> Result<Self> {
        tracing::info!(
            model_path = %model_path,
            session_len,
            tp,
            quant_policy,
            "Starting Python bridge subprocess"
        );

        // Find the bridge script
        // Look in ../lmdeploy/turbomind/ relative to lmdeploy-rust-server
        let bridge_script = std::path::PathBuf::from("../lmdeploy/turbomind/python_bridge.py");
        let bridge_script = if !bridge_script.exists() {
            // Try relative to project root
            std::path::PathBuf::from("../../lmdeploy/turbomind/python_bridge.py")
        } else {
            bridge_script
        };

        if !bridge_script.exists() {
            return Err(crate::error::AppError::ModelLoadFailed(format!(
                "Python bridge script not found: {:?}",
                bridge_script
            )));
        }

        // Start the Python subprocess
        let mut child = Command::new("python3")
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
            .map_err(|e| crate::error::AppError::ModelLoadFailed(format!("Failed to start Python bridge: {}", e)))?;

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
        let json = serde_json::to_string(cmd)
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("JSON encode error: {}", e)))?;

        let mut stdin = self.stdin.lock()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Lock error: {}", e)))?;
        writeln!(stdin, "{}", json)
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Write error: {}", e)))?;
        stdin.flush()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Flush error: {}", e)))?;
        drop(stdin);

        self.read_response()
    }

    /// Read a response from the bridge
    fn read_response(&self) -> Result<BridgeResponse> {
        let mut stdout = self.stdout.lock()
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Lock error: {}", e)))?;
        let mut line = String::new();
        stdout.read_line(&mut line)
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("Read error: {}", e)))?;
        drop(stdout);

        let response: BridgeResponse = serde_json::from_str(&line)
            .map_err(|e| crate::error::AppError::InferenceFailed(format!("JSON decode error: {}", e)))?;

        if response.status != "ok" {
            return Err(crate::error::AppError::InferenceFailed(
                response.message.unwrap_or_else(|| "Unknown error".to_string())
            ));
        }

        Ok(response)
    }

    /// Generate tokens from input_ids, returning (output_ids, elapsed_ms)
    pub fn generate_with_metrics(&self, input_ids: Vec<u32>, max_new_tokens: usize) -> Result<(Vec<u32>, f64)> {
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

    /// Shutdown the bridge
    pub fn shutdown(&self) -> Result<()> {
        let _ = self.send_command(&BridgeCommand::Shutdown);

        let mut child = self.child.lock()
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
