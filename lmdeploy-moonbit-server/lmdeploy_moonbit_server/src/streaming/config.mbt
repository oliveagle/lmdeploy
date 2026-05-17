// src/streaming/config.mbt
// Streaming configuration for LMDeploy MoonBit API
// Reference: lmdeploy-rust-server/config/default.toml

/// Streaming configuration
pub struct StreamingConfig {
  /// Stream timeout in seconds (default: 600)
  stream_timeout_secs: Int
  /// SSE keep-alive interval in milliseconds (default: 30000)
  keepalive_interval_ms: Int
  /// Maximum tokens per chunk (default: 8)
  max_chunk_tokens: Int
  /// Enable first token latency tracking (default: true)
  enable_first_token_metrics: Bool
  /// Connection timeout in seconds (default: 30)
  connection_timeout_secs: Int
}

pub fn StreamingConfig::default() -> StreamingConfig {
  {
    stream_timeout_secs: 600,
    keepalive_interval_ms: 30000,
    max_chunk_tokens: 8,
    enable_first_token_metrics: true,
    connection_timeout_secs: 30,
  }
}

use username/lmdeploy_moonbit_server/config

/// Create streaming config from ServerConfig (integrated config system)
pub fn StreamingConfig::from_server_config(sc: config.ServerConfig) -> StreamingConfig {
  {
    stream_timeout_secs: sc.stream_timeout_secs(),
    keepalive_interval_ms: sc.stream_keepalive_interval_ms(),
    max_chunk_tokens: sc.stream_max_chunk_tokens(),
  }
}

/// Create streaming config from environment variables
pub fn StreamingConfig::from_env() -> StreamingConfig {
  let default = StreamingConfig::default()

  {
    stream_timeout_secs: get_env_int("LMDEPLOY_STREAM_TIMEOUT_SECS", default.stream_timeout_secs),
    keepalive_interval_ms: get_env_int("LMDEPLOY_KEEPALIVE_INTERVAL_MS", default.keepalive_interval_ms),
    max_chunk_tokens: get_env_int("LMDEPLOY_MAX_CHUNK_TOKENS", default.max_chunk_tokens),
    enable_first_token_metrics: get_env_bool("LMDEPLOY_ENABLE_FIRST_TOKEN_METRICS", default.enable_first_token_metrics),
    connection_timeout_secs: get_env_int("LMDEPLOY_CONNECTION_TIMEOUT_SECS", default.connection_timeout_secs),
  }
}

/// Get integer from environment with default
fn get_env_int(key: String, default: Int) -> Int {
  // Requires WASI getenv FFI
  default
}

/// Get boolean from environment with default
fn get_env_bool(key: String, default: Bool) -> Bool {
  // Requires WASI getenv FFI
  default
}

/// SSE event types
pub enum SseEventType {
  /// Regular chunk event
  Chunk
  /// Keep-alive comment
  KeepAlive
  /// Stream termination
  Done
  /// Error event
  Error
}

pub fn SseEventType::to_string(self : SseEventType) -> String {
  match self {
    Chunk => "chat.completion.chunk"
    KeepAlive => ""
    Done => ""
    Error => "error"
  }
}
