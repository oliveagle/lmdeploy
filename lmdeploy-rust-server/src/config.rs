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
}

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
