//! Splintr tokenizer wrapper for LMDeploy
//!
//! Loads tokenizers from model directories (tokenizer.json)
//! using splintr for high-performance encoding/decoding.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;
use splintr::core::byte_level::byte_level_decode;
use splintr::Tokenizer;

/// LMDeploy tokenizer wrapper backed by splintr
#[derive(Clone)]
pub struct LMTokenizer {
    tokenizer: Arc<Tokenizer>,
    bos_token_id: Option<u32>,
    eos_token_ids: Vec<u32>,
    vocab_size: usize,
}

impl LMTokenizer {
    /// Create a new tokenizer from a model directory.
    pub fn from_path(model_dir: &str) -> Result<Self> {
        let path = Path::new(model_dir);

        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists() {
            let (tokenizer, bos, eos) = load_bpe_from_json(&tokenizer_json)?;
            return Self::new(tokenizer, bos, eos);
        }

        Err(anyhow!(
            "No tokenizer files found in {}. Expected tokenizer.json",
            model_dir
        ))
    }

    fn new(
        tokenizer: Tokenizer,
        bos_token_id: Option<u32>,
        eos_token_ids: Vec<u32>,
    ) -> Result<Self> {
        let vocab_size = tokenizer.vocab_size();
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            bos_token_id,
            eos_token_ids,
            vocab_size,
        })
    }

    pub fn encode(&self, text: &str, add_bos: bool, _add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut token_ids = self.tokenizer.encode(text);

        if add_bos {
            if let Some(bos_id) = self.bos_token_id {
                token_ids.insert(0, bos_id);
            }
        }

        if token_ids.len() >= 2 {
            if let Some(bos_id) = self.bos_token_id {
                if token_ids[0] == bos_id && token_ids[1] == bos_id {
                    tracing::warn!(
                        "Detected duplicate bos token {} in prompt, removing one",
                        bos_id
                    );
                    token_ids.remove(0);
                }
            }
        }

        Ok(token_ids)
    }

    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text))
    }

    pub fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids)
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.tokenizer
            .decode(&[token_id])
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        let bytes = self.tokenizer.decoder().get(&token_id)?;
        String::from_utf8(bytes.clone()).ok()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub fn encode_batch(&self, texts: &[&str], add_bos: bool) -> Result<Vec<Vec<u32>>> {
        let string_texts: Vec<String> = texts.iter().map(|&t| t.to_string()).collect();
        let results = self.tokenizer.encode_batch(&string_texts);

        let mut output = Vec::with_capacity(results.len());
        for mut token_ids in results {
            if add_bos {
                if let Some(bos_id) = self.bos_token_id {
                    token_ids.insert(0, bos_id);
                }
            }
            if token_ids.len() >= 2 {
                if let Some(bos_id) = self.bos_token_id {
                    if token_ids[0] == bos_id && token_ids[1] == bos_id {
                        tracing::warn!(
                            "Detected duplicate bos token {} in prompt, removing one",
                            bos_id
                        );
                        token_ids.remove(0);
                    }
                }
            }
            output.push(token_ids);
        }
        Ok(output)
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

// ============================================================================
// Tokenizer loading: parse tokenizer.json -> splintr Tokenizer
// ============================================================================

/// Load HuggingFace tokenizer.json (byte-level BPE, e.g. Qwen).
fn load_bpe_from_json(path: &Path) -> Result<(Tokenizer, Option<u32>, Vec<u32>)> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("Failed to read {}: {}", path.display(), e))?;

    let parsed: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| anyhow!("Failed to parse tokenizer.json: {}", e))?;

    let model = parsed
        .get("model")
        .ok_or_else(|| anyhow!("Missing 'model' field"))?;

    let model_type = model
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing model type"))?;

    if model_type != "BPE" {
        return Err(anyhow!("Unsupported model type: {}", model_type));
    }

    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("Missing vocab"))?;

    // Detect byte-level decoder
    let is_byte_level = model
        .get("decoder")
        .and_then(|d| d.get("type"))
        .and_then(|v| v.as_str())
        == Some("ByteLevel");

    // Build encoder: bytes -> token ID
    let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    for (token_str, id_val) in vocab {
        let id = id_val.as_u64().ok_or_else(|| anyhow!("Invalid token ID"))? as u32;
        let bytes = if is_byte_level {
            byte_level_decode(token_str)
                .ok_or_else(|| anyhow!("Byte-level decode failed for token: {:?}", token_str))?
        } else {
            token_str.as_bytes().to_vec()
        };
        encoder.insert(bytes, id);
    }

    // Collect special tokens from added_tokens
    let mut special_tokens: FxHashMap<String, u32> = FxHashMap::default();
    let mut bos_token_id: Option<u32> = None;
    let mut eos_token_ids: Vec<u32> = Vec::new();

    if let Some(added_tokens) = parsed.get("added_tokens").and_then(|v| v.as_array()) {
        for entry in added_tokens {
            let id = match entry.get("id").and_then(|v| v.as_u64()) {
                Some(v) => v as u32,
                None => continue,
            };
            let content = match entry.get("content").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => continue,
            };
            let is_special = entry
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_special {
                special_tokens.insert(content.to_string(), id);
            }

            match content {
                "<s>" | "<bos>" | "<BOS>" => bos_token_id = Some(id),
                "</s>" | "<eos>" | "<EOS>" => eos_token_ids.push(id),
                _ => {}
            }
        }
    }

    // Fallback: scan vocab
    if bos_token_id.is_none() {
        if let Some(id) = find_bpe_token(&encoder, is_byte_level, "<s>") {
            bos_token_id = Some(id);
        }
    }
    if eos_token_ids.is_empty() {
        if let Some(id) = find_bpe_token(&encoder, is_byte_level, "</s>") {
            eos_token_ids.push(id);
        }
    }

    let pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+| ?(?:\r\n\s*|\s+)|\s+(?!\S)|\s+";

    let tokenizer = if is_byte_level {
        Tokenizer::new_byte_level(encoder, special_tokens, pattern)
            .map_err(|e| anyhow!("Failed to create ByteLevel tokenizer: {}", e))?
    } else {
        Tokenizer::new(encoder, special_tokens, pattern)
            .map_err(|e| anyhow!("Failed to create tokenizer: {}", e))?
    };

    Ok((tokenizer, bos_token_id, eos_token_ids))
}

fn find_bpe_token(
    encoder: &FxHashMap<Vec<u8>, u32>,
    is_byte_level: bool,
    token: &str,
) -> Option<u32> {
    if is_byte_level {
        let bytes = byte_level_decode(token)?;
        encoder.get(&bytes).copied()
    } else {
        encoder.get(token.as_bytes()).copied()
    }
}
