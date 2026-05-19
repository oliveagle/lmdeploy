use lmdeploy_server::config::AppConfig;
use lmdeploy_server::error::AppError;
use lmdeploy_server::server::start_server;

use std::path::Path;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

#[tokio::main]
async fn main() -> Result<(), AppError> {
    // Parse command line args
    let config_path = std::env::args()
        .skip(1)
        .find(|arg| !arg.starts_with('-'))
        .filter(|p| Path::new(p).exists());

    let config = if let Some(path) = config_path {
        tracing::info!(config_path = %path, "Loading config from file");
        AppConfig::load_from_file(&path).map_err(|e| AppError::Config(e.to_string()))?
    } else {
        AppConfig::load().map_err(|e| AppError::Config(e.to_string()))?
    };

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(&config.logging.level)
    });

    let fmt_layer = match config.logging.json_format {
        true => tracing_subscriber::fmt::layer().json().with_filter(env_filter).boxed(),
        false => tracing_subscriber::fmt::layer().with_filter(env_filter).boxed(),
    };

    tracing_subscriber::registry().with(fmt_layer).init();

    tracing::info!(
        config = ?config,
        "Starting LMDeploy Rust API Server"
    );

    start_server(&config).await
}
