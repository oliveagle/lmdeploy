//! Model engine implementation
//!
//! NOTE: This module contains mock implementations of the TurboMind engine.
//! The current implementations return placeholder data and are not connected
//! to the actual TurboMind inference engine. These mocks serve as placeholders
//! for the API surface and infrastructure.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::Result;

/// Model loading state
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ModelState {
    /// Model is not loaded
    #[default]
    Unloaded,
    /// Model is currently loading
    Loading,
    /// Model is loaded and ready
    Ready,
    /// Model loading failed
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

pub struct TurboMindEngine {
    pub model_path: String,
    pub model_name: String,
    pub state: ModelState,
    pub loaded_at: Option<i64>,
    /// Flag to track if model is ready for inference
    is_ready: AtomicBool,
}

impl TurboMindEngine {
    pub async fn new(model_path: &str) -> Result<Self> {
        tracing::info!(model_path, "Initializing TurboMind engine (mock)");

        Ok(Self {
            model_path: model_path.to_string(),
            model_name: std::path::Path::new(model_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("default")
                .to_string(),
            state: ModelState::Ready,
            loaded_at: Some(unix_timestamp()),
            is_ready: AtomicBool::new(true),
        })
    }

    /// Check if the model is ready for inference
    pub fn is_ready(&self) -> bool {
        self.is_ready.load(Ordering::Relaxed)
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

    /// Reload the model (simulate hot reload)
    pub async fn reload(&mut self, new_model_path: &str) -> Result<()> {
        tracing::info!(
            old_path = %self.model_path,
            new_path = %new_model_path,
            "Reloading TurboMind engine"
        );

        self.is_ready.store(false, Ordering::Relaxed);
        self.state = ModelState::Loading;

        // Simulate reload delay
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        self.model_path = new_model_path.to_string();
        self.model_name = std::path::Path::new(new_model_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string();
        self.loaded_at = Some(unix_timestamp());
        self.state = ModelState::Ready;
        self.is_ready.store(true, Ordering::Relaxed);

        tracing::info!(
            model_path = %self.model_path,
            model_name = %self.model_name,
            "TurboMind engine reloaded successfully"
        );

        Ok(())
    }

    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        tracing::info!(prompt_len = prompt.len(), max_tokens, "TurboMind generate");

        format!("Mock response for: {}", &prompt[..prompt.len().min(50)])
    }

    pub async fn generate_stream(&self, prompt: &str) -> impl futures::Stream<Item = String> {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let mut stream = VecDeque::new();

        for word in words {
            stream.push_back(format!("{} ", word));
        }
        stream.push_back("[END]".to_string());

        futures::stream::iter(stream)
    }

    /// Generate embeddings for the given text
    /// Returns a vector of floats representing the embedding
    ///
    /// NOTE: This is a mock implementation that generates deterministic pseudo-random vectors.
    /// In production, this would call the actual TurboMind embedding model.
    pub async fn embed(&self, text: &str, dimensions: Option<usize>) -> Vec<f32> {
        tracing::info!(text_len = text.len(), "TurboMind embed");

        // Use requested dimensions or default to 1536 (OpenAI ada-002 compatible)
        let embedding_dim = dimensions.unwrap_or(1536);
        let mut embedding = Vec::with_capacity(embedding_dim);

        // Generate deterministic embedding based on text hash
        let mut hash: u64 = 5381;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }

        for i in 0..embedding_dim {
            // Use hash to generate deterministic values in [-1, 1]
            let val = ((hash.wrapping_mul(i as u64 + 1) % 2000000) as f32 / 1000000.0) - 1.0;
            embedding.push(val);
        }

        // Normalize the embedding (L2 normalization)
        let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
        let norm = sum_sq.sqrt();
        if norm > 0.0 {
            for val in embedding.iter_mut() {
                *val /= norm;
            }
        }

        embedding
    }
}

fn unix_timestamp() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
