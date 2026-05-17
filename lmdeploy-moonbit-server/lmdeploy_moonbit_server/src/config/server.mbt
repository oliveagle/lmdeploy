// src/config/server.mbt
// Server configuration with HTTP/2, connection pool, streaming, batching settings
// Reference: lmdeploy-rust-server/src/config.rs ServerConfig

/// Server configuration
pub struct ServerConfig {
  // HTTP/HTTPS settings
  http_addr: String
  http_port: Int

  // gRPC settings
  grpc_addr: String
  grpc_port: Int

  // Worker threads
  workers: Int

  // HTTP/2 settings
  http2_enabled: Bool
  http2_keepalive_interval_secs: Int
  http2_keepalive_timeout_secs: Int

  // Connection pool settings
  max_connections: Int
  connection_timeout_secs: Int
  request_timeout_secs: Int

  // Streaming settings
  stream_timeout_secs: Int
  stream_keepalive_interval_ms: Int
  stream_max_chunk_tokens: Int

  // Batching settings
  batch_enabled: Bool
  batch_size: Int
  batch_timeout_ms: Int

  // Rate limiting settings
  rate_limit_enabled: Bool
  rate_limit_max_requests: Int
  rate_limit_window_secs: Int

  // Graceful shutdown
  shutdown_timeout_secs: Int
}

pub fn ServerConfig::default() -> ServerConfig {
  {
    http_addr: "0.0.0.0",
    http_port: 3000,
    grpc_addr: "0.0.0.0",
    grpc_port: 50051,
    workers: 4,
    http2_enabled: true,
    http2_keepalive_interval_secs: 60,
    http2_keepalive_timeout_secs: 10,
    max_connections: 10000,
    connection_timeout_secs: 30,
    request_timeout_secs: 300,
    stream_timeout_secs: 600,
    stream_keepalive_interval_ms: 30000,
    stream_max_chunk_tokens: 8,
    batch_enabled: true,
    batch_size: 8,
    batch_timeout_ms: 50,
    rate_limit_enabled: true,
    rate_limit_max_requests: 100,
    rate_limit_window_secs: 60,
    shutdown_timeout_secs: 30,
  }
}

pub fn ServerConfig::new(
  http_addr: String,
  http_port: Int,
  workers: Int,
) -> ServerConfig {
  let defaults = ServerConfig::default()
  {
    http_addr,
    http_port,
    grpc_addr: defaults.grpc_addr,
    grpc_port: defaults.grpc_port,
    workers,
    http2_enabled: defaults.http2_enabled,
    http2_keepalive_interval_secs: defaults.http2_keepalive_interval_secs,
    http2_keepalive_timeout_secs: defaults.http2_keepalive_timeout_secs,
    max_connections: defaults.max_connections,
    connection_timeout_secs: defaults.connection_timeout_secs,
    request_timeout_secs: defaults.request_timeout_secs,
    stream_timeout_secs: defaults.stream_timeout_secs,
    stream_keepalive_interval_ms: defaults.stream_keepalive_interval_ms,
    stream_max_chunk_tokens: defaults.stream_max_chunk_tokens,
    batch_enabled: defaults.batch_enabled,
    batch_size: defaults.batch_size,
    batch_timeout_ms: defaults.batch_timeout_ms,
    rate_limit_enabled: defaults.rate_limit_enabled,
    rate_limit_max_requests: defaults.rate_limit_max_requests,
    rate_limit_window_secs: defaults.rate_limit_window_secs,
    shutdown_timeout_secs: defaults.shutdown_timeout_secs,
  }
}

/// Get HTTP port
pub fn ServerConfig::http_port(self: ServerConfig) -> Int {
  self.http_port
}

/// Get gRPC port
pub fn ServerConfig::grpc_port(self: ServerConfig) -> Int {
  self.grpc_port
}

/// Get HTTP bind address
pub fn ServerConfig::http_address(self: ServerConfig) -> String {
  self.http_addr ++ ":" ++ int_to_string(self.http_port)
}

/// Get gRPC bind address
pub fn ServerConfig::grpc_address(self: ServerConfig) -> String {
  self.grpc_addr ++ ":" ++ int_to_string(self.grpc_port)
}

/// Get streaming timeout in seconds
pub fn ServerConfig::stream_timeout_secs(self: ServerConfig) -> Int {
  self.stream_timeout_secs
}

/// Get streaming keepalive interval in milliseconds
pub fn ServerConfig::stream_keepalive_interval_ms(self: ServerConfig) -> Int {
  self.stream_keepalive_interval_ms
}

/// Get max chunk tokens for streaming
pub fn ServerConfig::stream_max_chunk_tokens(self: ServerConfig) -> Int {
  self.stream_max_chunk_tokens
}

/// Get max connections
pub fn ServerConfig::max_connections(self: ServerConfig) -> Int {
  self.max_connections
}

/// Get connection timeout in seconds
pub fn ServerConfig::connection_timeout_secs(self: ServerConfig) -> Int {
  self.connection_timeout_secs
}

/// Get request timeout in seconds
pub fn ServerConfig::request_timeout_secs(self: ServerConfig) -> Int {
  self.request_timeout_secs
}

/// Get worker thread count
pub fn ServerConfig::workers(self: ServerConfig) -> Int {
  self.workers
}

/// Check if HTTP/2 is enabled
pub fn ServerConfig::http2_enabled(self: ServerConfig) -> Bool {
  self.http2_enabled
}

/// Check if batching is enabled
pub fn ServerConfig::batch_enabled(self: ServerConfig) -> Bool {
  self.batch_enabled
}

/// Get batch size
pub fn ServerConfig::batch_size(self: ServerConfig) -> Int {
  self.batch_size
}

/// Get batch timeout in milliseconds
pub fn ServerConfig::batch_timeout_ms(self: ServerConfig) -> Int {
  self.batch_timeout_ms
}

/// Get shutdown timeout in seconds
pub fn ServerConfig::shutdown_timeout_secs(self: ServerConfig) -> Int {
  self.shutdown_timeout_secs
}

/// Rate limiting enabled
pub fn ServerConfig::rate_limit_enabled(self: ServerConfig) -> Bool {
  self.rate_limit_enabled
}

/// Max requests per window
pub fn ServerConfig::rate_limit_max_requests(self: ServerConfig) -> Int {
  self.rate_limit_max_requests
}

/// Rate limit window in seconds
pub fn ServerConfig::rate_limit_window_secs(self: ServerConfig) -> Int {
  self.rate_limit_window_secs
}

/// Parse from JSON string
fn ServerConfig::from_json(json: String) -> ServerConfig {
  let defaults = ServerConfig::default()
  {
    http_addr: parse_json_string_field(json, "http_addr", defaults.http_addr),
    http_port: parse_json_int_field(json, "http_port", defaults.http_port),
    grpc_addr: parse_json_string_field(json, "grpc_addr", defaults.grpc_addr),
    grpc_port: parse_json_int_field(json, "grpc_port", defaults.grpc_port),
    workers: parse_json_int_field(json, "workers", defaults.workers),
    http2_enabled: parse_json_bool_field(json, "http2_enabled", defaults.http2_enabled),
    http2_keepalive_interval_secs: parse_json_int_field(json, "http2_keepalive_interval_secs", defaults.http2_keepalive_interval_secs),
    http2_keepalive_timeout_secs: parse_json_int_field(json, "http2_keepalive_timeout_secs", defaults.http2_keepalive_timeout_secs),
    max_connections: parse_json_int_field(json, "max_connections", defaults.max_connections),
    connection_timeout_secs: parse_json_int_field(json, "connection_timeout_secs", defaults.connection_timeout_secs),
    request_timeout_secs: parse_json_int_field(json, "request_timeout_secs", defaults.request_timeout_secs),
    stream_timeout_secs: parse_json_int_field(json, "stream_timeout_secs", defaults.stream_timeout_secs),
    stream_keepalive_interval_ms: parse_json_int_field(json, "stream_keepalive_interval_ms", defaults.stream_keepalive_interval_ms),
    stream_max_chunk_tokens: parse_json_int_field(json, "stream_max_chunk_tokens", defaults.stream_max_chunk_tokens),
    batch_enabled: parse_json_bool_field(json, "batch_enabled", defaults.batch_enabled),
    batch_size: parse_json_int_field(json, "batch_size", defaults.batch_size),
    batch_timeout_ms: parse_json_int_field(json, "batch_timeout_ms", defaults.batch_timeout_ms),
    rate_limit_enabled: parse_json_bool_field(json, "rate_limit_enabled", defaults.rate_limit_enabled),
    rate_limit_max_requests: parse_json_int_field(json, "rate_limit_max_requests", defaults.rate_limit_max_requests),
    rate_limit_window_secs: parse_json_int_field(json, "rate_limit_window_secs", defaults.rate_limit_window_secs),
    shutdown_timeout_secs: parse_json_int_field(json, "shutdown_timeout_secs", defaults.shutdown_timeout_secs),
  }
}

/// Apply environment variable overrides
/// Environment variable format: LMDEPLOY_SERVER_* (e.g., LMDEPLOY_SERVER_HTTP_PORT)
fn ServerConfig::from_env_overrides(base: ServerConfig) -> ServerConfig {
  {
    http_addr: get_env_string("LMDEPLOY_SERVER_HTTP_ADDR", base.http_addr),
    http_port: get_env_int("LMDEPLOY_SERVER_HTTP_PORT", base.http_port),
    grpc_addr: get_env_string("LMDEPLOY_SERVER_GRPC_ADDR", base.grpc_addr),
    grpc_port: get_env_int("LMDEPLOY_SERVER_GRPC_PORT", base.grpc_port),
    workers: get_env_int("LMDEPLOY_SERVER_WORKERS", base.workers),
    http2_enabled: get_env_bool("LMDEPLOY_SERVER_HTTP2_ENABLED", base.http2_enabled),
    http2_keepalive_interval_secs: get_env_int("LMDEPLOY_SERVER_HTTP2_KEEPALIVE_INTERVAL_SECS", base.http2_keepalive_interval_secs),
    http2_keepalive_timeout_secs: get_env_int("LMDEPLOY_SERVER_HTTP2_KEEPALIVE_TIMEOUT_SECS", base.http2_keepalive_timeout_secs),
    max_connections: get_env_int("LMDEPLOY_SERVER_MAX_CONNECTIONS", base.max_connections),
    connection_timeout_secs: get_env_int("LMDEPLOY_SERVER_CONNECTION_TIMEOUT_SECS", base.connection_timeout_secs),
    request_timeout_secs: get_env_int("LMDEPLOY_SERVER_REQUEST_TIMEOUT_SECS", base.request_timeout_secs),
    stream_timeout_secs: get_env_int("LMDEPLOY_SERVER_STREAM_TIMEOUT_SECS", base.stream_timeout_secs),
    stream_keepalive_interval_ms: get_env_int("LMDEPLOY_SERVER_STREAM_KEEPALIVE_INTERVAL_MS", base.stream_keepalive_interval_ms),
    stream_max_chunk_tokens: get_env_int("LMDEPLOY_SERVER_STREAM_MAX_CHUNK_TOKENS", base.stream_max_chunk_tokens),
    batch_enabled: get_env_bool("LMDEPLOY_SERVER_BATCH_ENABLED", base.batch_enabled),
    batch_size: get_env_int("LMDEPLOY_SERVER_BATCH_SIZE", base.batch_size),
    batch_timeout_ms: get_env_int("LMDEPLOY_SERVER_BATCH_TIMEOUT_MS", base.batch_timeout_ms),
    rate_limit_enabled: get_env_bool("LMDEPLOY_SERVER_RATE_LIMIT_ENABLED", base.rate_limit_enabled),
    rate_limit_max_requests: get_env_int("LMDEPLOY_SERVER_RATE_LIMIT_MAX_REQUESTS", base.rate_limit_max_requests),
    rate_limit_window_secs: get_env_int("LMDEPLOY_SERVER_RATE_LIMIT_WINDOW_SECS", base.rate_limit_window_secs),
    shutdown_timeout_secs: get_env_int("LMDEPLOY_SERVER_SHUTDOWN_TIMEOUT_SECS", base.shutdown_timeout_secs),
  }
}

pub fn ServerConfig::to_debug_string(self: ServerConfig) -> String {
  "ServerConfig {"
    ++ " http_addr: \"" ++ self.http_addr ++ "\""
    ++ ", http_port: " ++ int_to_string(self.http_port)
    ++ ", grpc_addr: \"" ++ self.grpc_addr ++ "\""
    ++ ", grpc_port: " ++ int_to_string(self.grpc_port)
    ++ ", workers: " ++ int_to_string(self.workers)
    ++ ", http2_enabled: " ++ bool_to_string(self.http2_enabled)
    ++ ", batch_enabled: " ++ bool_to_string(self.batch_enabled)
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
    // Simplified for larger numbers
    let first = i / 1000
    int_to_string(first) ++ "000"
  }
}

fn bool_to_string(b: Bool) -> String {
  if b { "true" } else { "false" }
}

fn parse_json_string_field(json: String, field: String, default: String) -> String {
  // Simple JSON parsing - look for "field": "value"
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
  // Simple JSON parsing - look for "field": value
  let pattern = "\"" ++ field ++ "\":"
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    // Skip whitespace
    let mut start = start
    while start < json.length() && (json[start] == ' ' || json[start] == '\t' || json[start] == '\n') {
      start = start + 1
    }
    // Find end (comma or closing brace)
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
  // Simple JSON parsing - look for "field": true/false
  let pattern = "\"" ++ field ++ "\":"
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    // Check for "true" or "false"
    let true_pattern = "true"
    let false_pattern = "false"
    if json.find_from(true_pattern, start) == start {
      true
    } else if json.find_from(false_pattern, start) == start {
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

fn get_env_int(key: String, default: Int) -> Int {
  // Requires WASI getenv FFI. Returns default until FFI binding is available.
  default
}

fn get_env_bool(key: String, default: Bool) -> Bool {
  // Requires WASI getenv FFI. Returns default until FFI binding is available.
  default
}

fn string_to_int(s: String) -> Int {
  // Simple string to int conversion
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
