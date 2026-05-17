// src/config/logging.mbt
// Logging configuration
// Reference: lmdeploy-rust-server/src/config.rs LoggingConfig

/// Logging configuration
pub struct LoggingConfig {
  level: String
  json_format: Bool
}

pub fn LoggingConfig::default() -> LoggingConfig {
  {
    level: "info",
    json_format: true,
  }
}

pub fn LoggingConfig::new(level: String, json_format: Bool) -> LoggingConfig {
  let normalized = normalize_level(level)
  { level: normalized, json_format }
}

fn normalize_level(level: String) -> String {
  match level {
    "debug" | "DEBUG" | "Debug" => "debug"
    "info" | "INFO" | "Info" => "info"
    "warn" | "WARN" | "Warn" | "warning" | "WARNING" => "warn"
    "error" | "ERROR" | "Error" => "error"
    _ => "info"
  }
}

fn LoggingConfig::from_json(json: String) -> LoggingConfig {
  let defaults = LoggingConfig::default()
  {
    level: parse_json_string_field(json, "level", defaults.level),
    json_format: parse_json_bool_field(json, "json_format", defaults.json_format),
  }
}

fn LoggingConfig::from_env_overrides(base: LoggingConfig) -> LoggingConfig {
  {
    level: get_env_string("LMDEPLOY_LOG_LEVEL", base.level),
    json_format: get_env_bool("LMDEPLOY_LOG_JSON_FORMAT", base.json_format),
  }
}

pub fn LoggingConfig::to_debug_string(self: LoggingConfig) -> String {
  "LoggingConfig { level: \"" ++ self.level ++ "\", json_format: " ++ bool_to_string(self.json_format) ++ " }"
}

pub fn LoggingConfig::is_debug(self: LoggingConfig) -> Bool {
  self.level == "debug"
}

pub fn LoggingConfig::is_info(self: LoggingConfig) -> Bool {
  self.level == "info" || self.level == "debug"
}

// Helper functions

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
  // Requires WASI getenv FFI. Returns default until FFI binding is available.
  default
}

fn get_env_bool(key: String, default: Bool) -> Bool {
  // Requires WASI getenv FFI. Returns default until FFI binding is available.
  default
}

fn bool_to_string(b: Bool) -> String {
  if b { "true" } else { "false" }
}
