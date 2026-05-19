//! HuggingFace Tokenizer wrapper for LMDeploy
//!
//! Loads tokenizers from model directories (tokenizer.json, tokenizer.model, etc.)
//! and provides encode/decode interfaces for the TurboMind engine.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tokenizers::Tokenizer;

/// LMDeploy tokenizer wrapper
#[derive(Clone)]
pub struct LMTokenizer {
    tokenizer: Arc<Tokenizer>,
    /// BOS token ID
    bos_token_id: Option<u32>,
    /// EOS token IDs (may have multiple for chat models)
    eos_token_ids: Vec<u32>,
    /// Vocab size
    vocab_size: usize,
}

impl LMTokenizer {
    /// Create a new tokenizer from a model directory
    ///
    /// Searches for tokenizer files in priority order:
    /// 1. tokenizer.json (preferred - fast tokenizer format)
    /// 2. tokenizer.model (SentencePiece format)
    pub fn from_path(model_dir: &str) -> Result<Self> {
        let path = Path::new(model_dir);

        // Try loading tokenizer.json first
        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists() {
            let tokenizer = Tokenizer::from_file(&tokenizer_json)
                .map_err(|e| anyhow!("Failed to load tokenizer.json: {}", e))?;
            return Self::from_tokenizer(tokenizer);
        }

        // Try tokenizer.model (SentencePiece)
        let tokenizer_model = path.join("tokenizer.model");
        if tokenizer_model.exists() {
            let tokenizer = Tokenizer::from_file(&tokenizer_model)
                .map_err(|e| anyhow!("Failed to load tokenizer.model: {}", e))?;
            return Self::from_tokenizer(tokenizer);
        }

        // Also try tokenizer_config.json for vocab_file paths
        let tokenizer_config = path.join("tokenizer_config.json");
        if tokenizer_config.exists() {
            // Parse config to find vocab file
            if let Ok(config_str) = std::fs::read_to_string(&tokenizer_config) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                    // Try to find vocab_file path
                    if let Some(vocab_path) = config.get("vocab_file").and_then(|v| v.as_str()) {
                        let full_path = if Path::new(vocab_path).is_absolute() {
                            Path::new(vocab_path).to_path_buf()
                        } else {
                            path.join(vocab_path)
                        };
                        if full_path.exists() {
                            let tokenizer = Tokenizer::from_file(&full_path)
                                .map_err(|e| anyhow!("Failed to load vocab_file: {}", e))?;
                            return Self::from_tokenizer(tokenizer);
                        }
                    }
                }
            }
        }

        Err(anyhow!(
            "No tokenizer files found in {}. Expected tokenizer.json, tokenizer.model, or tokenizer_config.json",
            model_dir
        ))
    }

    /// Create from an existing Tokenizer instance
    fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        let vocab = tokenizer.get_vocab(true);
        let vocab_size = vocab.len();

        // Get BOS token ID (try multiple common names)
        let bos_token_id = tokenizer.token_to_id("<s>")
            .or_else(|| tokenizer.token_to_id("<bos>"))
            .or_else(|| tokenizer.token_to_id("<BOS>"));

        // Get EOS token IDs (try multiple common names)
        let mut eos_token_ids = Vec::new();
        for token in &["</s>", "<eos>", "<EOS>", "<|endoftext|>", "[EOS]"] {
            if let Some(id) = tokenizer.token_to_id(token) {
                eos_token_ids.push(id);
            }
        }

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            bos_token_id,
            eos_token_ids,
            vocab_size,
        })
    }

    /// Encode a string into token IDs
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `add_bos` - Whether to prepend the BOS token
    /// * `add_special_tokens` - Whether to add special tokens (handled by tokenizer)
    pub fn encode(&self, text: &str, add_bos: bool, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("Encoding failed: {}", e))?;

        let mut token_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();

        if add_bos {
            if let Some(bos_id) = self.bos_token_id {
                token_ids.insert(0, bos_id);
            }
        }

        Ok(token_ids)
    }

    /// Encode without any special token handling (raw tokenization)
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Encoding failed: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&id| id as u32).collect())
    }

    /// Decode token IDs back to string
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens in output
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let ids: Vec<u32> = token_ids.to_vec();
        let text = self.tokenizer
            .decode(&ids, skip_special_tokens)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;
        Ok(text)
    }

    /// Decode a single token ID
    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        let text = self.tokenizer
            .decode(&[token_id], true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;
        Ok(text)
    }

    /// Convert token ID to string (without decoding context)
    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.tokenizer.id_to_token(token_id)
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get the EOS token IDs
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Check if a token ID is an EOS token
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    /// Get batch encode - encodes multiple texts at once for efficiency
    pub fn encode_batch(&self, texts: &[&str], add_bos: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("Batch encoding failed: {}", e))?;

        let mut results = Vec::with_capacity(encodings.len());
        for encoding in encodings {
            let mut token_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();
            if add_bos {
                if let Some(bos_id) = self.bos_token_id {
                    token_ids.insert(0, bos_id);
                }
            }
            results.push(token_ids);
        }
        Ok(results)
    }
}

impl std::fmt::Debug for LMTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LMTokenizer")
            .field("vocab_size", &self.vocab_size)
            .field("bos_token_id", &self.bos_token_id)
            .field("eos_token_ids", &self.eos_token_ids)
            .finish()
    }
}