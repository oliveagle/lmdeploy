use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};

const DEFAULT_CONFIG_PATH: &str = "/etc/lmdeploy/config.toml";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub cache: CacheConfig,
    pub logging: LoggingConfig,
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub multi_model: MultiModelConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub http_addr: String,
    pub http_port: u16,
    pub grpc_addr: String,
    pub grpc_port: u16,
    pub workers: usize,
    // HTTP/2 settings
    #[serde(default = "default_http2_enabled")]
    pub http2_enabled: bool,
    #[serde(default = "default_http2_keepalive_interval")]
    pub http2_keepalive_interval_secs: u64,
    #[serde(default = "default_http2_keepalive_timeout")]
    pub http2_keepalive_timeout_secs: u64,
    // Connection pool settings
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_connection_timeout")]
    pub connection_timeout_secs: u64,
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
    // Streaming settings
    #[serde(default = "default_stream_timeout")]
    pub stream_timeout_secs: u64,
    #[serde(default = "default_stream_keepalive")]
    pub stream_keepalive_interval_ms: u64,
    #[serde(default = "default_stream_chunk_size")]
    pub stream_max_chunk_tokens: usize,
    // Batching settings
    #[serde(default = "default_batch_enabled")]
    pub batch_enabled: bool,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_batch_timeout")]
    pub batch_timeout_ms: u64,
    // Graceful shutdown
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout_secs: u64,
    // Error handling and rate limiting
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
    #[serde(default = "default_default_timeout_secs")]
    pub default_timeout_secs: u64,
}

fn default_http2_enabled() -> bool { true }
fn default_http2_keepalive_interval() -> u64 { 60 }
fn default_http2_keepalive_timeout() -> u64 { 10 }
fn default_max_connections() -> usize { 10000 }
fn default_connection_timeout() -> u64 { 30 }
fn default_request_timeout() -> u64 { 300 }
fn default_stream_timeout() -> u64 { 600 }
fn default_stream_keepalive() -> u64 { 30000 }
fn default_stream_chunk_size() -> usize { 8 }
fn default_batch_enabled() -> bool { true }
fn default_batch_size() -> usize { 8 }
fn default_batch_timeout() -> u64 { 50 }
fn default_shutdown_timeout() -> u64 { 30 }
fn default_default_timeout_secs() -> u64 { 60 }

/// Rate limiting configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    #[serde(default = "default_rate_limit_enabled")]
    pub enabled: bool,
    /// Requests per second (global)
    #[serde(default = "default_requests_per_second")]
    pub requests_per_second: u32,
    /// Burst size (allows temporary spikes)
    #[serde(default = "default_burst_size")]
    pub burst_size: u32,
    /// Per-IP rate limiting
    #[serde(default)]
    pub per_ip: PerIpRateLimitConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerIpRateLimitConfig {
    /// Enable per-IP rate limiting
    #[serde(default)]
    pub enabled: bool,
    /// Requests per second per IP
    #[serde(default = "default_per_ip_requests_per_second")]
    pub requests_per_second: u32,
    /// Burst size per IP
    #[serde(default = "default_per_ip_burst_size")]
    pub burst_size: u32,
    /// Maximum number of tracked IPs
    #[serde(default = "default_max_tracked_ips")]
    pub max_tracked_ips: usize,
}

fn default_rate_limit_enabled() -> bool { false }
fn default_requests_per_second() -> u32 { 100 }
fn default_burst_size() -> u32 { 200 }
fn default_per_ip_requests_per_second() -> u32 { 30 }
fn default_per_ip_burst_size() -> u32 { 60 }
fn default_max_tracked_ips() -> usize { 10000 }

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: 100,
            burst_size: 200,
            per_ip: PerIpRateLimitConfig::default(),
        }
    }
}

impl Default for PerIpRateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: 30,
            burst_size: 60,
            max_tracked_ips: 10000,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub model_name: String,
    pub max_context_length: usize,
    pub max_batch_size: usize,
    /// Engine type: "python_bridge" (default) or "pure_cpp"
    /// Python bridge is more compatible, pure_cpp has no Python dependency
    #[serde(default = "default_engine_type")]
    pub engine_type: String,
}

fn default_engine_type() -> String {
    "python_bridge".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheConfig {
    pub tokenizer_cache_size: usize,
    pub tokenizer_ttl_secs: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub json_format: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MetricsConfig {
    #[serde(default = "default_metrics_enabled")]
    pub enabled: bool,
    #[serde(default = "default_metrics_addr")]
    pub host: String,
    #[serde(default = "default_metrics_port")]
    pub port: u16,
}

fn default_metrics_enabled() -> bool { true }
fn default_metrics_addr() -> String { "0.0.0.0".into() }
fn default_metrics_port() -> u16 { 9090 }

/// Additional model to load (for multi-model support)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
    /// Engine type: "python_bridge" (default) or "pure_cpp"
    #[serde(default = "default_model_engine_type")]
    pub engine_type: String,
}

fn default_model_engine_type() -> String {
    "python_bridge".to_string()
}

/// Multi-model deployment configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct MultiModelConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                http_addr: "0.0.0.0".into(),
                http_port: 3000,
                grpc_addr: "0.0.0.0".into(),
                grpc_port: 50051,
                workers: num_cpus::get(),
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
                shutdown_timeout_secs: 30,
                rate_limit: RateLimitConfig::default(),
                default_timeout_secs: 60,
            },
            model: ModelConfig {
                model_path: "".into(),
                model_name: "default".into(),
                max_context_length: 8192,
                max_batch_size: 8,
                engine_type: default_engine_type(),
            },
            cache: CacheConfig {
                tokenizer_cache_size: 1000,
                tokenizer_ttl_secs: 3600,
            },
            logging: LoggingConfig {
                level: "info".into(),
                json_format: true,
            },
            metrics: MetricsConfig {
                enabled: true,
                host: "0.0.0.0".into(),
                port: 9090,
            },
            multi_model: MultiModelConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Result<Self, config::ConfigError> {
        let mut builder = Config::builder()
            .add_source(File::with_name(DEFAULT_CONFIG_PATH).required(false))
            .add_source(Environment::with_prefix("LMDEPLOY").separator("_"));

        builder = builder.add_source(
            config::File::from_str(
                include_str!("../config/default.toml"),
                config::FileFormat::Toml,
            )
            .required(false),
        );

        let mut config: Self = builder.build()?.try_deserialize()?;

        // Override model_path from LMDEPLOY_MODEL_PATH if set
        // (the standard config crate expects LMDEPLOY_MODEL_MODEL_PATH for nested fields)
        if let Ok(path) = std::env::var("LMDEPLOY_MODEL_PATH") {
            if !path.is_empty() {
                config.model.model_path = path;
            }
        }

        Ok(config)
    }
}
