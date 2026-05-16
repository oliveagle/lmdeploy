// src/config/metrics.mbt
// Metrics configuration for Prometheus
// Reference: lmdeploy-rust-server/src/config.rs MetricsConfig

/// Metrics configuration
pub struct MetricsConfig {
  enabled: Bool
  host: String
  port: Int
}

pub fn MetricsConfig::default() -> MetricsConfig {
  {
    enabled: true,
    host: "0.0.0.0",
    port: 9090,
  }
}

pub fn MetricsConfig::new(enabled: Bool, host: String, port: Int) -> MetricsConfig {
  { enabled, host, port }
}

fn MetricsConfig::from_json(json: String) -> MetricsConfig {
  let defaults = MetricsConfig::default()
  {
    enabled: parse_json_bool_field(json, "enabled", defaults.enabled),
    host: parse_json_string_field(json, "host", defaults.host),
    port: parse_json_int_field(json, "port", defaults.port),
  }
}

fn MetricsConfig::from_env_overrides(base: MetricsConfig) -> MetricsConfig {
  {
    enabled: get_env_bool("LMDEPLOY_METRICS_ENABLED", base.enabled),
    host: get_env_string("LMDEPLOY_METRICS_HOST", base.host),
    port: get_env_int("LMDEPLOY_METRICS_PORT", base.port),
  }
}

pub fn MetricsConfig::address(self: MetricsConfig) -> String {
  self.host ++ ":" ++ int_to_string(self.port)
}

pub fn MetricsConfig::is_enabled(self: MetricsConfig) -> Bool {
  self.enabled
}

pub fn MetricsConfig::disable(self: MetricsConfig) -> MetricsConfig {
  { enabled: false, host: self.host, port: self.port }
}

pub fn MetricsConfig::enable(self: MetricsConfig) -> MetricsConfig {
  { enabled: true, host: self.host, port: self.port }
}

pub fn MetricsConfig::to_debug_string(self: MetricsConfig) -> String {
  "MetricsConfig { enabled: " ++ bool_to_string(self.enabled)
    ++ ", host: \"" ++ self.host ++ "\""
    ++ ", port: " ++ int_to_string(self.port) ++ " }"
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
    if json.find_from("true", start) == start {
      true
    } else if json.find_from("false", start) == start {
      false
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

fn get_env_bool(key: String, default: Bool) -> Bool {
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

fn bool_to_string(b: Bool) -> String {
  if b { "true" } else { "false" }
}
