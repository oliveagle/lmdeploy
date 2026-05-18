//! TurboMind Engine - Python Bridge integration
//!
//! Uses the Python TurboMind API via a bridge subprocess for inference.
//! This bypasses the broken C API weight loading for AWQ models.
//!
//! # Model Format Support
//!
//! Supports HuggingFace safetensors models including AWQ quantization.
//! The Python bridge handles all weight loading and inference.
//!
//! # AWQ Quantization Support
//!
//! AWQ (Activation-Aware Weight Quantization) models are automatically detected
//! from config.json and passed to the Python bridge with appropriate quant_policy.

use std::sync::Arc;

use crate::error::Result;
use crate::model::python_bridge::PythonBridge;
use crate::tokenizer::LMTokenizer;

/// Model loading state
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ModelState {
    #[default]
    Unloaded,
    Loading,
    Ready,
    Failed(String),
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub state: ModelState,
    pub loaded_at: Option<i64>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            name: "default".into(),
            path: "".into(),
            state: ModelState::Unloaded,
            loaded_at: None,
        }
    }
}

/// Detect AWQ quantization from config.json
fn detect_awq_quantization(model_path: &std::path::Path) -> bool {
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return false;
    }

    std::fs::read_to_string(&config_path)
        .ok()
        .map(|content| {
            let content_lower = content.to_lowercase();
            content_lower.contains("\"quant_method\"") && content_lower.contains("\"awq\"")
        })
        .unwrap_or(false)
}

/// TurboMind engine using Python bridge
pub struct TurboMindEngine {
    pub(super) model_path: String,
    pub model_name: String,
    state: ModelState,
    loaded_at: Option<i64>,
    is_ready: std::sync::atomic::AtomicBool,

    // Python bridge for inference
    bridge: Option<Arc<PythonBridge>>,

    // Tokenizer for encoding/decoding
    tokenizer: Option<LMTokenizer>,
}

impl TurboMindEngine {
    /// Create a new engine and initialize via Python bridge
    pub async fn new(model_path: &str) -> Result<Self> {
        tracing::info!(model_path, "Initializing TurboMind engine via Python bridge");

        let model_path_obj = std::path::PathBuf::from(model_path);
        let config_json = model_path_obj.join("config.json");

        if !config_json.exists() {
            tracing::error!(model_path = %model_path, "config.json not found in model path");
            return Err(crate::error::AppError::ModelLoadFailed(
                format!("config.json not found at {}", model_path)
            ));
        }

        // Detect AWQ quantization
        let is_awq = detect_awq_quantization(&model_path_obj);
        let quant_policy = if is_awq { 4 } else { 0 };

        if is_awq {
            tracing::info!("Detected AWQ quantized model, enabling quant_policy=4");
        }

        // Load tokenizer first (for text encode/decode)
        tracing::info!("Loading tokenizer...");
        let tokenizer = match LMTokenizer::from_path(model_path) {
            Ok(t) => {
                tracing::info!(vocab_size = t.vocab_size(), "Tokenizer loaded successfully");
                Some(t)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load tokenizer from model path");
                None
            }
        };

        // Create Python bridge (which loads the model)
        tracing::info!("Starting Python bridge subprocess...");
        let bridge = PythonBridge::new(
            model_path,
            2048,  // session_len
            1,     // tp
            quant_policy,
        )?;

        let model_name = model_path_obj
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string();

        Ok(Self {
            model_path: model_path.to_string(),
            model_name,
            state: ModelState::Ready,
            loaded_at: Some(unix_timestamp()),
            is_ready: std::sync::atomic::AtomicBool::new(true),
            bridge: Some(Arc::new(bridge)),
            tokenizer,
        })
    }

    /// Check if the model is ready for inference
    pub fn is_ready(&self) -> bool {
        self.is_ready.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get model metadata
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            path: self.model_path.clone(),
            state: self.state.clone(),
            loaded_at: self.loaded_at,
        }
    }

    /// Reload the model
    pub async fn reload(&mut self, new_model_path: &str) -> Result<()> {
        tracing::info!(
            old_path = %self.model_path,
            new_path = %new_model_path,
            "Reloading TurboMind engine"
        );

        self.is_ready.store(false, std::sync::atomic::Ordering::Relaxed);
        self.state = ModelState::Loading;

        // Drop existing bridge
        self.bridge = None;

        self.model_path = new_model_path.to_string();
        self.model_name = std::path::Path::new(new_model_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string();

        // Re-initialize
        let new_engine = Self::new(new_model_path).await?;
        self.bridge = new_engine.bridge;
        self.state = ModelState::Ready;
        self.loaded_at = Some(unix_timestamp());
        self.is_ready.store(true, std::sync::atomic::Ordering::Relaxed);

        tracing::info!(
            model_path = %self.model_path,
            model_name = %self.model_name,
            "TurboMind engine reloaded successfully"
        );

        Ok(())
    }

    /// Generate text with TurboMind
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let bridge = self.bridge.as_ref().expect("Python bridge not initialized");

        // Tokenize input
        let input_ids = if let Some(tokenizer) = &self.tokenizer {
            match tokenizer.encode(prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed");
                    return String::new();
                }
            }
        } else {
            tracing::error!("Tokenizer not available");
            return String::new();
        };

        tracing::debug!(input_len = input_ids.len(), "Tokenized prompt");

        // Generate via Python bridge
        let output_ids = match bridge.generate(input_ids, max_tokens) {
            Ok(ids) => ids,
            Err(e) => {
                tracing::error!(error = %e, "Generation failed");
                return String::new();
            }
        };

        // Decode output
        if let Some(tokenizer) = &self.tokenizer {
            match tokenizer.decode(&output_ids, true) {
                Ok(text) => text,
                Err(e) => {
                    tracing::error!(error = %e, "Decoding failed");
                    String::new()
                }
            }
        } else {
            String::new()
        }
    }

    /// Generate text with streaming (not implemented for Python bridge yet)
    pub async fn generate_stream(&self, _prompt: &str) -> std::pin::Pin<Box<dyn futures::Stream<Item = String> + Send>> {
        tracing::warn!("Streaming not implemented for Python bridge");
        Box::pin(futures::stream::empty())
    }

    /// Generate embeddings (not supported)
    pub async fn embed(&self, _text: &str, _dimensions: Option<usize>) -> Vec<f32> {
        tracing::warn!("Embeddings not supported by Python bridge");
        Vec::new()
    }
}

fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the initialization sequence state machine
    #[tokio::test]
    async fn test_state_transitions() {
        let state_unloaded = ModelState::Unloaded;
        let state_loading = ModelState::Loading;
        let state_ready = ModelState::Ready;
        let state_failed = ModelState::Failed("test error".to_string());

        assert_eq!(state_unloaded, ModelState::Unloaded);
        assert_eq!(state_loading, ModelState::Loading);
        assert_eq!(state_ready, ModelState::Ready);
        assert!(matches!(state_failed, ModelState::Failed(msg) if msg == "test error"));
    }

    /// Test ModelInfo structure
    #[tokio::test]
    async fn test_model_info() {
        let info = ModelInfo {
            name: "test-model".to_string(),
            path: "/path/to/model".to_string(),
            state: ModelState::Ready,
            loaded_at: Some(1234567890),
        };

        assert_eq!(info.name, "test-model");
        assert_eq!(info.path, "/path/to/model");
        assert_eq!(info.state, ModelState::Ready);
        assert_eq!(info.loaded_at, Some(1234567890));
    }
}
