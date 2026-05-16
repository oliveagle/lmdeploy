use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};

const DEFAULT_CONFIG_PATH: &str = "/etc/lmdeploy/config.toml";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub cache: CacheConfig,
    pub logging: LoggingConfig,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub model_name: String,
    pub max_context_length: usize,
    pub max_batch_size: usize,
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
            },
            model: ModelConfig {
                model_path: "".into(),
                model_name: "default".into(),
                max_context_length: 8192,
                max_batch_size: 8,
            },
            cache: CacheConfig {
                tokenizer_cache_size: 1000,
                tokenizer_ttl_secs: 3600,
            },
            logging: LoggingConfig {
                level: "info".into(),
                json_format: true,
            },
        }
    }
}

impl AppConfig {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
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

        let config = builder.build()?.try_deserialize()?;
        Ok(config)
    }
}
