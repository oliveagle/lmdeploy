// src/router/router.mbt
// HTTP router for OpenAI-compatible API endpoints
// Reference: lmdeploy/serve/openai/api_server.py route definitions

use username/lmdeploy_moonbit_server/http
use username/lmdeploy_moonbit_server/handlers
use username/lmdeploy_moonbit_server/config

/// Router configuration
pub struct RouterConfig {
  api_version: String
  enable_batch: Bool
  enable_metrics: Bool
  enable_streaming: Bool
  enable_model_management: Bool
}

pub fn RouterConfig::default() -> RouterConfig {
  {
    api_version: "v1",
    enable_batch: true,
    enable_metrics: true,
    enable_streaming: true,
    enable_model_management: true,
  }
}

/// HTTP router for LMDeploy API
pub struct ApiRouter {
  config: RouterConfig
  server: http.Http2Server
}

pub fn ApiRouter::new(config: RouterConfig) -> ApiRouter {
  let http_config = http.Http2Config::default()
  let server = http.Http2Server::new(http_config)

  let router = ApiRouter::setup_routes({
    config,
    server,
  })

  router
}

/// Setup all routes for the API
fn ApiRouter::setup_routes(self : ApiRouter) -> ApiRouter {
  let mut server = self.server

  // Health check endpoint (not versioned)
  server = server.add_route(http.HttpMethod::Get, "/health", handlers::health_check_handler)
  server = server.add_route(http.HttpMethod::Get, "/health/", handlers::health_check_handler)

  // Metrics endpoint (not versioned)
  if self.config.enable_metrics {
    server = server.add_route(http.HttpMethod::Get, "/metrics", handlers::metrics_handler)
  }

  // Model endpoints
  server = server.add_route(http.HttpMethod::Get, "/" ++ self.config.api_version ++ "/models", handlers::list_models_handler)

  // Chat completion endpoints
  server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/chat/completions", handlers::chat_completions_handler)

  // Completion endpoints
  server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/completions", handlers::completions_handler)

  // Embeddings endpoint
  server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/embeddings", handlers::embeddings_handler)

  // Encode endpoint (tokenization)
  server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/encode", handlers::encode_handler)

  // Pooling endpoint
  server = server.add_route(http.HttpMethod::Post, "/pooling", handlers::pooling_handler)

  // Batch processing endpoints
  if self.config.enable_batch {
    server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/chat/completions/batch", handlers::batch_chat_completions_handler)
    server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/completions/batch", handlers::batch_completions_handler)
  }

  // Model management endpoints
  if self.config.enable_model_management {
    server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/models/load", handlers::load_model_handler)
    server = server.add_route(http.HttpMethod::Post, "/" ++ self.config.api_version ++ "/models/unload", handlers::unload_model_handler)
    server = server.add_route(http.HttpMethod::Get, "/" ++ self.config.api_version ++ "/models/progress", handlers::model_progress_handler)
  }

  { config: self.config, server }
}

/// Get the underlying HTTP server
pub fn ApiRouter::server(self : ApiRouter) -> http.Http2Server {
  self.server
}

/// Start the router
pub fn ApiRouter::start(self : ApiRouter) -> Result[http.Http2Server, String] {
  self.server.start()
}

/// Stop the router
pub fn ApiRouter::stop(self : ApiRouter) -> ApiRouter {
  {
    config: self.config,
    server: self.server.stop(),
  }
}
