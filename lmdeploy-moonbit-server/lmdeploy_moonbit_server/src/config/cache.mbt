// src/config/cache.mbt
// Cache configuration
// Reference: lmdeploy-rust-server/src/config.rs CacheConfig

/// Cache configuration
pub struct CacheConfig {
  tokenizer_cache_size: Int
  tokenizer_ttl_secs: Int
}

pub fn CacheConfig::default() -> CacheConfig {
  {
    tokenizer_cache_size: 1000,
    tokenizer_ttl_secs: 3600,
  }
}

pub fn CacheConfig::new(cache_size: Int, ttl_secs: Int) -> CacheConfig {
  { tokenizer_cache_size: cache_size, tokenizer_ttl_secs: ttl_secs }
}

/// Parse from JSON string
fn CacheConfig::from_json(json: String) -> CacheConfig {
  let defaults = CacheConfig::default()
  {
    tokenizer_cache_size: parse_json_int_field(json, "tokenizer_cache_size", defaults.tokenizer_cache_size),
    tokenizer_ttl_secs: parse_json_int_field(json, "tokenizer_ttl_secs", defaults.tokenizer_ttl_secs),
  }
}

/// Apply environment variable overrides
/// Environment variable format: LMDEPLOY_CACHE_* (e.g., LMDEPLOY_CACHE_TOKENIZER_CACHE_SIZE)
fn CacheConfig::from_env_overrides(base: CacheConfig) -> CacheConfig {
  {
    tokenizer_cache_size: get_env_int("LMDEPLOY_CACHE_TOKENIZER_CACHE_SIZE", base.tokenizer_cache_size),
    tokenizer_ttl_secs: get_env_int("LMDEPLOY_CACHE_TOKENIZER_TTL_SECS", base.tokenizer_ttl_secs),
  }
}

/// Get tokenizer cache size
pub fn CacheConfig::tokenizer_cache_size(self: CacheConfig) -> Int {
  self.tokenizer_cache_size
}

/// Get tokenizer TTL in seconds
pub fn CacheConfig::tokenizer_ttl_secs(self: CacheConfig) -> Int {
  self.tokenizer_ttl_secs
}

pub fn CacheConfig::to_debug_string(self: CacheConfig) -> String {
  "CacheConfig {"
    ++ " tokenizer_cache_size: " ++ int_to_string(self.tokenizer_cache_size)
    ++ ", tokenizer_ttl_secs: " ++ int_to_string(self.tokenizer_ttl_secs)
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
