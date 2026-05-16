// src/config/config.mbt
// Main configuration module for LMDeploy MoonBit API Server
// Reference: lmdeploy-rust-server/src/config.rs, LMDEPLOY_CLAUDE.md Python config
// Supports JSON/JSONC config files with environment variable overrides

mod model
mod cache
mod server
mod logging
mod metrics

/// Main application configuration
pub struct AppConfig {
  server: server.ServerConfig
  model: model.ModelConfig
  cache: cache.CacheConfig
  logging: logging.LoggingConfig
  metrics: metrics.MetricsConfig
}

pub fn AppConfig::default() -> AppConfig {
  {
    server: server.ServerConfig::default(),
    model: model.ModelConfig::default(),
    cache: cache.CacheConfig::default(),
    logging: logging.LoggingConfig::default(),
    metrics: metrics.MetricsConfig::default(),
  }
}

/// Default configuration path
pub const DEFAULT_CONFIG_PATH: String = "/etc/lmdeploy/config.jsonc"
pub const DEFAULT_CONFIG_PATH_ALT: String = "/etc/lmdeploy/config.json"

/// Load configuration with defaults, file, and env var overrides
/// Priority (lowest to highest): defaults → config file → env vars
pub fn AppConfig::load() -> AppConfig {
  let defaults = AppConfig::default()
  AppConfig::load_with_path(DEFAULT_CONFIG_PATH, defaults)
}

/// Load configuration from a specific path
pub fn AppConfig::load_with_path(path: String, defaults: AppConfig) -> AppConfig {
  let file_config = AppConfig::load_file(path)
  match file_config {
    Some(cfg) => AppConfig::from_file_with_env(cfg)
    None => AppConfig::from_env(defaults)
  }
}

/// Parse configuration from JSON string
pub fn AppConfig::from_json(json: String) -> AppConfig {
  let server = server.ServerConfig::from_json(json)
  let model = model.ModelConfig::from_json(json)
  let cache = cache.CacheConfig::from_json(json)
  let logging = logging.LoggingConfig::from_json(json)
  let metrics = metrics.MetricsConfig::from_json(json)
  { server, model, cache, logging, metrics }
}

/// Load config from file and apply env var overrides
fn AppConfig::from_file_with_env(file_cfg: AppConfig) -> AppConfig {
  let server = server.ServerConfig::from_env_overrides(file_cfg.server)
  let model = model.ModelConfig::from_env_overrides(file_cfg.model)
  let cache = cache.CacheConfig::from_env_overrides(file_cfg.cache)
  let logging = logging.LoggingConfig::from_env_overrides(file_cfg.logging)
  let metrics = metrics.MetricsConfig::from_env_overrides(file_cfg.metrics)
  { server, model, cache, logging, metrics }
}

/// Apply environment variable overrides to defaults
fn AppConfig::from_env(defaults: AppConfig) -> AppConfig {
  let server = server.ServerConfig::from_env_overrides(defaults.server)
  let model = model.ModelConfig::from_env_overrides(defaults.model)
  let cache = cache.CacheConfig::from_env_overrides(defaults.cache)
  let logging = logging.LoggingConfig::from_env_overrides(defaults.logging)
  let metrics = metrics.MetricsConfig::from_env_overrides(defaults.metrics)
  { server, model, cache, logging, metrics }
}

/// Try to load configuration from file
fn AppConfig::load_file(path: String) -> Option[AppConfig] {
  // Check primary path
  let exists = path_exists(path)
  if exists {
    let content = read_file_content(path)
    if content != "" {
      Some(AppConfig::from_json(content))
    } else {
      None
    }
  } else {
    // Check alternate path (.json vs .jsonc)
    let alt_path = if path.ends_with(".jsonc") {
      let len = path.length()
      path.substring(0, len - 1)
    } else {
      let len = path.length()
      path.substring(0, len - 4) ++ "c"
    }
    let alt_exists = path_exists(alt_path)
    if alt_exists {
      let content = read_file_content(alt_path)
      if content != "" {
        Some(AppConfig::from_json(content))
      } else {
        None
      }
    } else {
      None
    }
  }
}

pub fn AppConfig::server(self: AppConfig) -> server.ServerConfig {
  self.server
}

pub fn AppConfig::model(self: AppConfig) -> model.ModelConfig {
  self.model
}

pub fn AppConfig::cache(self: AppConfig) -> cache.CacheConfig {
  self.cache
}

pub fn AppConfig::logging(self: AppConfig) -> logging.LoggingConfig {
  self.logging
}

pub fn AppConfig::metrics(self: AppConfig) -> metrics.MetricsConfig {
  self.metrics
}

/// Print configuration as JSON-like string for debugging
pub fn AppConfig::to_debug_string(self: AppConfig) -> String {
  "{\n"
    ++ "  server: " ++ server.ServerConfig::to_debug_string(self.server) ++ "\n"
    ++ "  model: " ++ model.ModelConfig::to_debug_string(self.model) ++ "\n"
    ++ "  cache: " ++ cache.CacheConfig::to_debug_string(self.cache) ++ "\n"
    ++ "  logging: " ++ logging.LoggingConfig::to_debug_string(self.logging) ++ "\n"
    ++ "  metrics: " ++ metrics.MetricsConfig::to_debug_string(self.metrics) ++ "\n"
    ++ "}"
}

/// Check if a file path exists
/// TODO: Implement FFI to file system access
fn path_exists(path: String) -> Bool {
  false
}

/// Read file content as string
/// TODO: Implement FFI to file system access
fn read_file_content(path: String) -> String {
  ""
}
