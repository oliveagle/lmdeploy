use std::collections::VecDeque;

use crate::error::Result;

pub struct TurboMindEngine {
    pub model_path: String,
}

impl TurboMindEngine {
    pub async fn new(model_path: &str) -> Result<Self> {
        tracing::info!(model_path, "Initializing TurboMind engine (mock)");

        Ok(Self {
            model_path: model_path.to_string(),
        })
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
    pub async fn embed(&self, text: &str) -> Vec<f32> {
        tracing::info!(text_len = text.len(), "TurboMind embed");

        // Mock embedding: generate deterministic pseudo-random vector
        // In production, this would call the actual embedding model
        let embedding_dim = 1536; // Common embedding dimension (e.g., OpenAI ada-002)
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
