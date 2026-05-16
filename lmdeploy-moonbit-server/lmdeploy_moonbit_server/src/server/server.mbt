// src/server/server.mbt
// gRPC/HTTP server for LMDeploy MoonBit API
// Reference: lmdeploy-rust-server/src/server.rs

use username/lmdeploy_moonbit_server/config

/// Server configuration (wrapper for AppConfig)
pub struct ServerConfig {
  host: String
  port: Int
  grpc_port: Int
  workers: Int
  max_connections: Int
  request_timeout_ms: Int
  model_name: String
  model_path: String
}

/// Load server config from AppConfig (integrated config system)
pub fn ServerConfig::from_app_config(cfg: config.AppConfig) -> ServerConfig {
  {
    host: cfg.server.http_address(),
    port: cfg.server.http_port(),
    grpc_port: cfg.server.grpc_port(),
    workers: cfg.server.workers(),
    max_connections: cfg.server.max_connections(),
    request_timeout_ms: cfg.server.request_timeout_secs() * 1000,
    model_name: cfg.model.model_name(),
    model_path: cfg.model.model_path(),
  }
}

pub fn ServerConfig::default() -> ServerConfig {
  // Load from default AppConfig
  let app_cfg = config.AppConfig::default()
  ServerConfig::from_app_config(app_cfg)
}

/// LMDeploy server
pub struct LmDeployServer {
  config: ServerConfig
  service: LmDeployService
  start_time: Int64
}

pub fn LmDeployServer::new(config: ServerConfig) -> LmDeployServer {
  let service = LmDeployService::new(
    "0.1.0",
    config.model_name,
  )
  {
    config,
    service,
    start_time: current_timestamp(),
  }
}

/// Start the server
/// TODO: Implement actual server startup with moonbit-http or gRPC bindings
pub fn LmDeployServer::start(self: LmDeployServer) -> Result[Unit, String] {
  // Placeholder for server startup
  Ok(())
}

/// Stop the server gracefully
pub fn LmDeployServer::stop(self: LmDeployServer) -> Result[Unit, String] {
  // Placeholder for graceful shutdown
  Ok(())
}

/// Get server uptime in seconds
pub fn LmDeployServer::uptime(self : LmDeployServer) -> Int64 {
  current_timestamp() - self.start_time
}

/// Get the gRPC service
pub fn LmDeployServer::service(self : LmDeployServer) -> LmDeployService {
  self.service
}
