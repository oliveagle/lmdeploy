// src/config/model.mbt
// Model configuration
// Reference: lmdeploy-rust-server/src/config.rs ModelConfig
// Reference: lmdeploy/pytorch/config.py ModelConfig
// Reference: src/turbomind/models/model_weight.h ModelWeightConfig

use username/lmdeploy_moonbit_server.config.quantization

/// Model configuration - includes all parameters needed for model loading
pub struct ModelConfig {
  // Basic model info
  model_path: String
  model_name: String
  max_context_length: Int
  max_batch_size: Int

  // Full model config parsed from config.json
  num_layers: Int
  num_heads: Int
  num_kv_heads: Int
  vocab_size: Int
  hidden_size: Int
  head_dim: Int

  // Additional parameters for ModelWeightConfig
  intermediate_size: Int
  bos_token_id: Int
  eos_token_id: Int
  sliding_window: Int
  tie_word_embeddings: Bool
  use_mrope: Bool

  // Tensor parallelism
  tp_size: Int
  tp_rank: Int

  // Data type
  data_type: String

  // Quantization configuration
  quantization: quantization.QuantizationConfig
}

pub fn ModelConfig::default() -> ModelConfig {
  {
    model_path: "",
    model_name: "default",
    max_context_length: 8192,
    max_batch_size: 8,
    num_layers: 0,
    num_heads: 0,
    num_kv_heads: 0,
    vocab_size: 152064, // Default tokenizer vocab size
    hidden_size: 0,
    head_dim: 128, // Default, common for many models
    intermediate_size: 0,
    bos_token_id: 0,
    eos_token_id: 0,
    sliding_window: -1,
    tie_word_embeddings: false,
    use_mrope: false,
    tp_size: 1,
    tp_rank: 0,
    data_type: "float16",
    quantization: quantization.QuantizationConfig::default(),
  }
}

pub fn ModelConfig::new(model_path: String, model_name: String) -> ModelConfig {
  let cfg = ModelConfig::default()
  // Try to parse config.json from model path
  let full_config = parse_model_config_from_path(model_path)
  // Parse quantization config from JSON
  let quant_config = parse_quantization_from_path(model_path)
  {
    model_path,
    model_name,
    max_context_length: 8192,
    max_batch_size: 8,
    num_layers: full_config.0,
    num_heads: full_config.1,
    num_kv_heads: full_config.4,
    vocab_size: full_config.2,
    hidden_size: full_config.3,
    head_dim: if full_config.1 > 0 && full_config.3 > 0 { full_config.3 / full_config.1 } else { 128 },
    intermediate_size: full_config.5,
    bos_token_id: full_config.6,
    eos_token_id: full_config.7,
    sliding_window: -1,
    tie_word_embeddings: false,
    use_mrope: false,
    tp_size: 1,
    tp_rank: 0,
    data_type: "float16",
    quantization: quant_config,
  }
}

/// Parse model config from config.json file in model path
/// Returns (num_layers, num_heads, vocab_size, hidden_size, num_kv_heads, intermediate_size, bos_token_id, eos_token_id)
fn parse_model_config_from_path(model_path: String) -> (Int, Int, Int, Int, Int, Int, Int, Int) {
  if model_path == "" {
    return (0, 0, 152064, 0, 0, 0, 0, 0)
  }

  // Build config.json path (append /config.json if path doesn't end with it)
  let config_path = if model_path.ends_with("config.json") {
    model_path
  } else {
    let len = model_path.length()
    let sep = if model_path[len - 1] == '/' { "" } else { "/" }
    model_path ++ sep ++ "config.json"
  }

  // Try to read and parse the config file
  let json_content = read_model_config_json(config_path)
  parse_model_config_json(json_content)
}

/// Parse quantization config from config.json in model path
fn parse_quantization_from_path(model_path: String) -> quantization.QuantizationConfig {
  if model_path == "" {
    return quantization.QuantizationConfig::default()
  }

  let config_path = if model_path.ends_with("config.json") {
    model_path
  } else {
    let len = model_path.length()
    let sep = if model_path[len - 1] == '/' { "" } else { "/" }
    model_path ++ sep ++ "config.json"
  }

  let json_content = read_model_config_json(config_path)
  quantization.parse_quantization_config(json_content)
}

/// Parse model config from JSON content
/// Returns (num_layers, num_heads, vocab_size, hidden_size, num_kv_heads, intermediate_size, bos_token_id, eos_token_id)
fn parse_model_config_json(json: String) -> (Int, Int, Int, Int, Int, Int, Int, Int) {
  if json == "" {
    return (0, 0, 152064, 0, 0, 0, 0, 0)
  }

  let num_layers = parse_json_int_field(json, "num_hidden_layers", 0)
  let num_heads = parse_json_int_field(json, "num_attention_heads", 0)
  let vocab_size = parse_json_int_field(json, "vocab_size", 152064)
  let hidden_size = parse_json_int_field(json, "hidden_size", 0)
  let num_kv_heads = parse_json_int_field(json, "num_key_value_heads", 0)
  let intermediate_size = parse_json_int_field(json, "intermediate_size", 0)
  let bos_token_id = parse_json_int_field(json, "bos_token_id", 0)
  let eos_token_id = parse_json_int_field(json, "eos_token_id", 0)

  // Handle nested text_config for some models (e.g., Qwen2 with text_config block)
  let final_num_heads = if num_heads == 0 {
    let text_heads = num_kv_heads // For GQA
    if text_heads > 0 { text_heads } else {
      parse_nested_int_field(json, "num_attention_heads", 0)
    }
  } else { num_heads }

  let final_hidden_size = if hidden_size == 0 {
    parse_nested_int_field(json, "hidden_size", 0)
  } else { hidden_size }

  let final_vocab_size = if vocab_size == 152064 {
    let text_vocab = parse_nested_int_field(json, "vocab_size", 152064)
    if text_vocab != 152064 { text_vocab } else { vocab_size }
  } else { vocab_size }

  let final_num_kv_heads = if num_kv_heads == 0 {
    parse_nested_int_field(json, "num_key_value_heads", 0)
  } else { num_kv_heads }

  let final_intermediate_size = if intermediate_size == 0 {
    parse_nested_int_field(json, "intermediate_size", 0)
  } else { intermediate_size }

  let final_bos_token_id = if bos_token_id == 0 {
    parse_nested_int_field(json, "bos_token_id", 0)
  } else { bos_token_id }

  let final_eos_token_id = if eos_token_id == 0 {
    parse_nested_int_field(json, "eos_token_id", 0)
  } else { eos_token_id }

  (num_layers, final_num_heads, final_vocab_size, final_hidden_size, final_num_kv_heads, final_intermediate_size, final_bos_token_id, final_eos_token_id)
}

/// Parse a nested field inside a nested block (e.g., "text_config": { ... })
fn parse_nested_int_field(json: String, field: String, default: Int) -> Int {
  let pattern = "\"" ++ field ++ "\":"
  // Find first occurrence after a nested object start
  let idx = json.find(pattern)
  if idx < 0 {
    default
  } else {
    // Extract just the number around this occurrence
    let rest = json.substring(idx + pattern.length(), json.length())
    parse_number_only(rest, default)
  }
}

/// Read model config JSON from file path (requires WASI FFI)
fn read_model_config_json(config_path: String) -> String {
  // Stub - requires WASI filesystem FFI
  // Actual implementation will read file via WASI host call
  ""
}

/// Parse from JSON string
fn ModelConfig::from_json(json: String) -> ModelConfig {
  let defaults = ModelConfig::default()
  let full_config = parse_model_config_json(json)
  let head_dim = if full_config.1 > 0 && full_config.3 > 0 { full_config.3 / full_config.1 } else { 128 }
  {
    model_path: parse_json_string_field(json, "model_path", defaults.model_path),
    model_name: parse_json_string_field(json, "model_name", defaults.model_name),
    max_context_length: parse_json_int_field(json, "max_context_length", defaults.max_context_length),
    max_batch_size: parse_json_int_field(json, "max_batch_size", defaults.max_batch_size),
    num_layers: parse_json_int_field(json, "num_layers", full_config.0),
    num_heads: parse_json_int_field(json, "num_heads", full_config.1),
    num_kv_heads: full_config.4,
    vocab_size: parse_json_int_field(json, "vocab_size", full_config.2),
    hidden_size: parse_json_int_field(json, "hidden_size", full_config.3),
    head_dim: head_dim,
    intermediate_size: full_config.5,
    bos_token_id: full_config.6,
    eos_token_id: full_config.7,
    sliding_window: parse_json_int_field(json, "sliding_window", -1),
    tie_word_embeddings: parse_json_bool_field(json, "tie_word_embeddings", false),
    use_mrope: parse_json_bool_field(json, "use_mrope", false),
    tp_size: parse_json_int_field(json, "tp_size", 1),
    tp_rank: parse_json_int_field(json, "tp_rank", 0),
    data_type: parse_json_string_field(json, "data_type", defaults.data_type),
    quantization: quantization.parse_quantization_config(json),
  }
}

/// Apply environment variable overrides
/// Environment variable format: LMDEPLOY_MODEL_* (e.g., LMDEPLOY_MODEL_MODEL_PATH)
fn ModelConfig::from_env_overrides(base: ModelConfig) -> ModelConfig {
  {
    model_path: get_env_string("LMDEPLOY_MODEL_MODEL_PATH", base.model_path),
    model_name: get_env_string("LMDEPLOY_MODEL_MODEL_NAME", base.model_name),
    max_context_length: get_env_int("LMDEPLOY_MODEL_MAX_CONTEXT_LENGTH", base.max_context_length),
    max_batch_size: get_env_int("LMDEPLOY_MODEL_MAX_BATCH_SIZE", base.max_batch_size),
    num_layers: get_env_int("LMDEPLOY_MODEL_NUM_LAYERS", base.num_layers),
    num_heads: get_env_int("LMDEPLOY_MODEL_NUM_HEADS", base.num_heads),
    num_kv_heads: get_env_int("LMDEPLOY_MODEL_NUM_KV_HEADS", base.num_kv_heads),
    vocab_size: get_env_int("LMDEPLOY_MODEL_VOCAB_SIZE", base.vocab_size),
    hidden_size: get_env_int("LMDEPLOY_MODEL_HIDDEN_SIZE", base.hidden_size),
    head_dim: get_env_int("LMDEPLOY_MODEL_HEAD_DIM", base.head_dim),
    intermediate_size: get_env_int("LMDEPLOY_MODEL_INTERMEDIATE_SIZE", base.intermediate_size),
    bos_token_id: get_env_int("LMDEPLOY_MODEL_BOS_TOKEN_ID", base.bos_token_id),
    eos_token_id: get_env_int("LMDEPLOY_MODEL_EOS_TOKEN_ID", base.eos_token_id),
    sliding_window: get_env_int("LMDEPLOY_MODEL_SLIDING_WINDOW", base.sliding_window),
    tie_word_embeddings: base.tie_word_embeddings,
    use_mrope: base.use_mrope,
    tp_size: get_env_int("LMDEPLOY_MODEL_TP_SIZE", base.tp_size),
    tp_rank: get_env_int("LMDEPLOY_MODEL_TP_RANK", base.tp_rank),
    data_type: base.data_type,
    quantization: base.quantization,
  }
}

/// Get model name
pub fn ModelConfig::model_name(self: ModelConfig) -> String {
  self.model_name
}

/// Get model path
pub fn ModelConfig::model_path(self: ModelConfig) -> String {
  self.model_path
}

/// Get max context length
pub fn ModelConfig::max_context_length(self: ModelConfig) -> Int {
  self.max_context_length
}

/// Get max batch size
pub fn ModelConfig::max_batch_size(self: ModelConfig) -> Int {
  self.max_batch_size
}

/// Get number of layers
pub fn ModelConfig::num_layers(self: ModelConfig) -> Int {
  self.num_layers
}

/// Get number of attention heads
pub fn ModelConfig::num_heads(self: ModelConfig) -> Int {
  self.num_heads
}

/// Get vocabulary size
pub fn ModelConfig::vocab_size(self: ModelConfig) -> Int {
  self.vocab_size
}

/// Get hidden size
pub fn ModelConfig::hidden_size(self: ModelConfig) -> Int {
  self.hidden_size
}

/// Get head dimension
pub fn ModelConfig::head_dim(self: ModelConfig) -> Int {
  self.head_dim
}

/// Get number of key-value heads (for GQA/MQA)
pub fn ModelConfig::num_kv_heads(self: ModelConfig) -> Int {
  self.num_kv_heads
}

/// Get intermediate size (MLP hidden dimension)
pub fn ModelConfig::intermediate_size(self: ModelConfig) -> Int {
  self.intermediate_size
}

/// Get BOS token ID
pub fn ModelConfig::bos_token_id(self: ModelConfig) -> Int {
  self.bos_token_id
}

/// Get EOS token ID
pub fn ModelConfig::eos_token_id(self: ModelConfig) -> Int {
  self.eos_token_id
}

/// Get sliding window size (-1 means disabled)
pub fn ModelConfig::sliding_window(self: ModelConfig) -> Int {
  self.sliding_window
}

/// Check if word embeddings are tied
pub fn ModelConfig::tie_word_embeddings(self: ModelConfig) -> Bool {
  self.tie_word_embeddings
}

/// Check if model uses multi-rope
pub fn ModelConfig::use_mrope(self: ModelConfig) -> Bool {
  self.use_mrope
}

/// Get tensor parallel size
pub fn ModelConfig::tp_size(self: ModelConfig) -> Int {
  self.tp_size
}

/// Get tensor parallel rank
pub fn ModelConfig::tp_rank(self: ModelConfig) -> Int {
  self.tp_rank
}

/// Get data type
pub fn ModelConfig::data_type(self: ModelConfig) -> String {
  self.data_type
}

/// Get quantization configuration
pub fn ModelConfig::quantization(self: ModelConfig) -> quantization.QuantizationConfig {
  self.quantization
}

/// Check if model is quantized
pub fn ModelConfig::is_quantized(self: ModelConfig) -> Bool {
  self.quantization.is_enabled()
}

/// Get quantization method (awq, gptq, none)
pub fn ModelConfig::quantization_method(self: ModelConfig) -> String {
  self.quantization.quant_method()
}

pub fn ModelConfig::to_debug_string(self: ModelConfig) -> String {
  let quant_str = match self.quantization {
    quantization.QuantizationConfig::None => "none"
    quantization.QuantizationConfig::Awq(cfg) => "awq_" ++ int_to_string(cfg.bits()) ++ "bit"
    quantization.QuantizationConfig::Gptq(cfg) => "gptq_" ++ int_to_string(cfg.bits()) ++ "bit"
    quantization.QuantizationConfig::Other(method) => method
  }
  "ModelConfig {"
    ++ " model_path: \"" ++ self.model_path ++ "\""
    ++ ", model_name: \"" ++ self.model_name ++ "\""
    ++ ", max_context_length: " ++ int_to_string(self.max_context_length)
    ++ ", max_batch_size: " ++ int_to_string(self.max_batch_size)
    ++ ", num_layers: " ++ int_to_string(self.num_layers)
    ++ ", num_heads: " ++ int_to_string(self.num_heads)
    ++ ", num_kv_heads: " ++ int_to_string(self.num_kv_heads)
    ++ ", vocab_size: " ++ int_to_string(self.vocab_size)
    ++ ", hidden_size: " ++ int_to_string(self.hidden_size)
    ++ ", head_dim: " ++ int_to_string(self.head_dim)
    ++ ", intermediate_size: " ++ int_to_string(self.intermediate_size)
    ++ ", bos_token_id: " ++ int_to_string(self.bos_token_id)
    ++ ", eos_token_id: " ++ int_to_string(self.eos_token_id)
    ++ ", sliding_window: " ++ int_to_string(self.sliding_window)
    ++ ", tie_word_embeddings: " ++ bool_to_string(self.tie_word_embeddings)
    ++ ", use_mrope: " ++ bool_to_string(self.use_mrope)
    ++ ", tp_size: " ++ int_to_string(self.tp_size)
    ++ ", tp_rank: " ++ int_to_string(self.tp_rank)
    ++ ", data_type: \"" ++ self.data_type ++ "\""
    ++ ", quantization: \"" ++ quant_str ++ "\""
    ++ " }"
}

// Helper functions

fn int_to_string(i: Int) -> String {
  if i < 0 {
    "-" ++ int_to_string(-i)
  } else if i < 10 {
    match i {
      0 => "0"
      1 => "1"
      2 => "2"
      3 => "3"
      4 => "4"
      5 => "5"
      6 => "6"
      7 => "7"
      8 => "8"
      _ => "9"
    }
  } else if i < 100 {
    let tens = i / 10
    let ones = i - (tens * 10)
    int_to_string(tens) ++ int_to_string(ones)
  } else if i < 1000 {
    let hundreds = i / 100
    let rest = i - (hundreds * 100)
    int_to_string(hundreds) ++ int_to_string(rest)
  } else if i < 10000 {
    let thousands = i / 1000
    let rest = i - (thousands * 1000)
    int_to_string(thousands) ++ int_to_string(rest)
  } else {
    let first = i / 1000
    int_to_string(first) ++ "000"
  }
}

fn parse_json_string_field(json: String, field: String, default: String) -> String {
  let pattern = "\"" ++ field ++ "\":\""
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    let end_idx = json.find_from("\"", start)
    if end_idx >= 0 {
      json.substring(start, end_idx)
    } else {
      default
    }
  } else {
    default
  }
}

fn parse_json_int_field(json: String, field: String, default: Int) -> Int {
  let pattern = "\"" ++ field ++ "\":"
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    let mut start = start
    while start < json.length() && (json[start] == ' ' || json[start] == '\t' || json[start] == '\n') {
      start = start + 1
    }
    let end_idx = json.find_from(",", start)
    let end_idx2 = json.find_from("}", start)
    let end = if end_idx >= 0 && (end_idx2 < 0 || end_idx < end_idx2) { end_idx } else { end_idx2 }
    if end >= 0 {
      let num_str = json.substring(start, end)
      string_to_int(num_str)
    } else {
      default
    }
  } else {
    default
  }
}

fn parse_json_bool_field(json: String, field: String, default: Bool) -> Bool {
  let pattern = "\"" ++ field ++ "\":"
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    let mut start = start
    while start < json.length() && (json[start] == ' ' || json[start] == '\t' || json[start] == '\n') {
      start = start + 1
    }
    let end_idx = json.find_from(",", start)
    let end_idx2 = json.find_from("}", start)
    let end = if end_idx >= 0 && (end_idx2 < 0 || end_idx < end_idx2) { end_idx } else { end_idx2 }
    if end >= 0 {
      let value_str = json.substring(start, end)
      if value_str == "true" {
        true
      } else if value_str == "false" {
        false
      } else {
        default
      }
    } else {
      default
    }
  } else {
    default
  }
}

fn get_env_string(key: String, default: String) -> String {
  // Requires WASI getenv FFI
  default
}

fn get_env_int(key: String, default: Int) -> Int {
  // Requires WASI getenv FFI
  default
}

fn string_to_int(s: String) -> Int {
  let mut result = 0
  let mut idx = 0
  while idx < s.length() {
    let c = s[idx]
    if c >= '0' && c <= '9' {
      result = result * 10 + (c - '0')
    }
    idx = idx + 1
  }
  result
}

/// Parse only the leading number from text (stops at non-digit)
fn parse_number_only(text: String, default: Int) -> Int {
  let mut result = 0
  let mut idx = 0
  while idx < text.length() {
    let c = text[idx]
    if c >= '0' && c <= '9' {
      result = result * 10 + (c - '0')
      idx = idx + 1
    } else {
      break
    }
  }
  if idx == 0 { default } else { result }
}

fn bool_to_string(b: Bool) -> String {
  if b { "true" } else { "false" }
}