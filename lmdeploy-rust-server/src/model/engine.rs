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
}
