use axum::{
    http::Method,
    routing::{get, post},
    Router,
};
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    cors::{AllowHeaders, AllowOrigin, CorsLayer},
    trace::TraceLayer,
};

use std::net::SocketAddr;
use std::sync::Arc;

use crate::cache::TokenizeCache;
use crate::config::AppConfig;
use crate::error::Result;
use crate::handlers::http::{
    cache_metrics, chat_completions, chat_completions_stream, clear_cache, completions, health_check, list_models,
    tokenize,
};
use crate::model::TurboMindEngine;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<RwLock<TurboMindEngine>>,
    pub tokenizer_cache: Arc<TokenizeCache>,
}

pub async fn start_server(config: &AppConfig) -> Result<()> {
    let engine = Arc::new(RwLock::new(
        TurboMindEngine::new(&config.model.model_path).await?,
    ));

    let tokenizer_cache = Arc::new(TokenizeCache::new(
        config.cache.tokenizer_cache_size,
        config.cache.tokenizer_ttl_secs,
    ));

    tracing::info!(
        cache_size = config.cache.tokenizer_cache_size,
        cache_ttl_secs = config.cache.tokenizer_ttl_secs,
        "Tokenize cache initialized"
    );

    let state = Arc::new(AppState {
        engine,
        tokenizer_cache: tokenizer_cache.clone(),
    });

    let grpc_addr = format!("{}:{}", config.server.grpc_addr, config.server.grpc_port)
        .parse::<SocketAddr>()
        .unwrap();

    let http_addr = format!("{}:{}", config.server.http_addr, config.server.http_port)
        .parse::<SocketAddr>()
        .unwrap();

    let grpc_handle = tokio::spawn(async move {
        crate::grpc::start_grpc_server(grpc_addr, env!("CARGO_PKG_VERSION").to_string(), tokenizer_cache).await
    });

    let http_handle = tokio::spawn(async move {
        let app = create_router(state);
        let listener = tokio::net::TcpListener::bind(&http_addr).await.unwrap();
        tracing::info!(addr = %http_addr, "HTTP server listening");
        axum::serve(listener, app).await.unwrap();
        Ok::<(), crate::error::AppError>(())
    });

    let (http_res, grpc_res) = tokio::join!(http_handle, grpc_handle);

    if let Err(e) = http_res {
        tracing::error!("HTTP server error: {:?}", e);
    }

    if let Err(e) = grpc_res {
        tracing::error!("gRPC server error: {:?}", e);
    }

    Ok(())
}

fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::mirror_request())
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(AllowHeaders::any());

    Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions/stream", post(chat_completions_stream))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/cache/metrics", get(cache_metrics))
        .route("/v1/cache/clear", post(clear_cache))
        .layer(TraceLayer::new_for_http())
        .layer(ServiceBuilder::new().layer(cors))
        .with_state(state)
}
