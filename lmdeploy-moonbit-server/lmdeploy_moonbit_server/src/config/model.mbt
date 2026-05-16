// src/config/model.mbt
// Model configuration
// Reference: lmdeploy-rust-server/src/config.rs ModelConfig

/// Model configuration
pub struct ModelConfig {
  model_path: String
  model_name: String
  max_context_length: Int
  max_batch_size: Int
}

pub fn ModelConfig::default() -> ModelConfig {
  {
    model_path: "",
    model_name: "default",
    max_context_length: 8192,
    max_batch_size: 8,
  }
}

pub fn ModelConfig::new(model_path: String, model_name: String) -> ModelConfig {
  {
    model_path,
    model_name,
    max_context_length: 8192,
    max_batch_size: 8,
  }
}

/// Parse from JSON string
fn ModelConfig::from_json(json: String) -> ModelConfig {
  let defaults = ModelConfig::default()
  {
    model_path: parse_json_string_field(json, "model_path", defaults.model_path),
    model_name: parse_json_string_field(json, "model_name", defaults.model_name),
    max_context_length: parse_json_int_field(json, "max_context_length", defaults.max_context_length),
    max_batch_size: parse_json_int_field(json, "max_batch_size", defaults.max_batch_size),
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

pub fn ModelConfig::to_debug_string(self: ModelConfig) -> String {
  "ModelConfig {"
    ++ " model_path: \"" ++ self.model_path ++ "\""
    ++ ", model_name: \"" ++ self.model_name ++ "\""
    ++ ", max_context_length: " ++ int_to_string(self.max_context_length)
    ++ ", max_batch_size: " ++ int_to_string(self.max_batch_size)
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

fn get_env_string(key: String, default: String) -> String {
  // TODO: Implement FFI to getenv()
  default
}

fn get_env_int(key: String, default: Int) -> Int {
  // TODO: Implement FFI to getenv()
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