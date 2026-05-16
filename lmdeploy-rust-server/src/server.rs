use axum::{
    extract::State,
    http::{Method, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    signal,
    sync::{mpsc, oneshot, RwLock, Semaphore},
};
use tower::ServiceBuilder;
use tower_http::{
    cors::{AllowHeaders, AllowOrigin, CorsLayer},
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
};

use crate::cache::TokenizeCache;
use crate::config::AppConfig;
use crate::error::{AppError, Result};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::handlers::http::{
    batch_stats, cache_metrics, chat_completions, chat_completions_stream, clear_cache, completions, health_check,
    list_models, tokenize, ChatCompletionsRequest, ChatCompletionsResponse, Choice, Message, Usage,
};
use crate::model::TurboMindEngine;

/// Type alias for the batch sender channel
pub type BatchSender = mpsc::UnboundedSender<BatchItem>;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<RwLock<TurboMindEngine>>,
    pub tokenizer_cache: Arc<TokenizeCache>,
    pub config: AppConfig,
    pub request_semaphore: Arc<Semaphore>,
    pub batch_sender: Option<Arc<BatchSender>>,
    pub batch_stats: Arc<BatchStats>,
}

/// Batch request accumulator
struct BatchAccumulator {
    requests: Vec<BatchItem>,
    last_add: Instant,
}

pub struct BatchItem {
    pub req: ChatCompletionsRequest,
    pub tx: oneshot::Sender<ChatCompletionsResponse>,
}

#[derive(Debug, Serialize)]
pub struct BatchStats {
    pub batch_count: AtomicU64,
    pub total_requests: AtomicU64,
}

impl Clone for BatchStats {
    fn clone(&self) -> Self {
        Self {
            batch_count: AtomicU64::new(self.batch_count.load(Ordering::Relaxed)),
            total_requests: AtomicU64::new(self.total_requests.load(Ordering::Relaxed)),
        }
    }
}

impl BatchStats {
    pub fn new() -> Self {
        Self {
            batch_count: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
        }
    }

    pub fn batch_response(&self) -> BatchStatsResponse {
        let batch_count = self.batch_count.load(Ordering::Relaxed);
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        BatchStatsResponse {
            batch_count,
            total_requests,
            avg_batch_size: if batch_count > 0 {
                total_requests as f64 / batch_count as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct BatchStatsResponse {
    pub batch_count: u64,
    pub total_requests: u64,
    pub avg_batch_size: f64,
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

    let request_semaphore = Arc::new(Semaphore::new(config.server.max_connections));
    let batch_stats = Arc::new(BatchStats::new());

    let http_addr = format!("{}:{}", config.server.http_addr, config.server.http_port)
        .parse::<SocketAddr>()
        .unwrap();
    let grpc_addr = format!("{}:{}", config.server.grpc_addr, config.server.grpc_port)
        .parse::<SocketAddr>()
        .unwrap();

    // Initialize batch sender if batching is enabled
    let batch_sender = if config.server.batch_enabled {
        let (batch_tx, batch_rx) = mpsc::unbounded_channel::<BatchItem>();
        let batch_size = config.server.batch_size;
        let batch_timeout_ms = config.server.batch_timeout_ms;
        let batch_timeout = Duration::from_millis(config.server.batch_timeout_ms);
        let stats_clone = batch_stats.clone();

        let handle = tokio::spawn(async move {
            tracing::info!(
                batch_size,
                timeout_ms = batch_timeout_ms,
                "Batch processor started"
            );
            run_batch_processor(batch_rx, batch_size, batch_timeout, stats_clone).await
        });

        Some((Arc::new(batch_tx), handle))
    } else {
        None
    };

    let state = Arc::new(AppState {
        engine,
        tokenizer_cache: tokenizer_cache.clone(),
        config: config.clone(),
        request_semaphore,
        batch_sender: batch_sender.as_ref().map(|(tx, _)| tx.clone()),
        batch_stats,
    });

    // Spawn gRPC server
    let grpc_handle = tokio::spawn(async move {
        crate::grpc::start_grpc_server(grpc_addr, env!("CARGO_PKG_VERSION").to_string(), tokenizer_cache).await
    });

    // Spawn HTTP server
    let http_state = state.clone();
    let http_addr_cloned = http_addr;
    let http2_enabled = config.server.http2_enabled;
    let max_connections = config.server.max_connections;
    let http_handle = tokio::spawn(async move {
        let app = create_router(http_state);
        let listener = tokio::net::TcpListener::bind(&http_addr_cloned).await?;

        tracing::info!(
            addr = %http_addr_cloned,
            http2_enabled = http2_enabled,
            max_connections = max_connections,
            "HTTP server listening"
        );

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| AppError::Other(e.to_string()))
    });

    // Wait for servers or errors
    tokio::select! {
        res = http_handle => {
            if let Err(e) = res {
                tracing::error!("HTTP server task error: {:?}", e);
            }
        }
        res = grpc_handle => {
            if let Err(e) = res {
                tracing::error!("gRPC server task error: {:?}", e);
            }
        }
    }

    // Shutdown batch processor
    if let Some((_, handle)) = batch_sender {
        handle.abort();
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
        .route("/v1/batch/stats", get(batch_stats))
        .layer(TraceLayer::new_for_http())
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB limit
        .layer(ServiceBuilder::new().layer(cors))
        .with_state(state)
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Ctrl+C received, starting graceful shutdown");
        },
        _ = terminate => {
            tracing::info!("TERM signal received, starting graceful shutdown");
        },
    }
}

/// Batch processor for accumulating and executing inference requests
async fn run_batch_processor(
    mut rx: mpsc::UnboundedReceiver<BatchItem>,
    batch_size: usize,
    batch_timeout: Duration,
    stats: Arc<BatchStats>,
) {
    let mut accumulator: HashMap<String, BatchAccumulator> = HashMap::new();

    loop {
        tokio::select! {
            // Receive new request
            item_opt = rx.recv() => {
                let Some(item) = item_opt else {
                    // Channel closed, flush remaining batches
                    flush_all_batches(&mut accumulator, &stats).await;
                    break;
                };

                let model = item.req.model.clone();
                let entry = accumulator.entry(model.clone()).or_insert_with(|| BatchAccumulator {
                    requests: Vec::new(),
                    last_add: Instant::now(),
                });

                entry.requests.push(item);
                stats.total_requests.fetch_add(1, Ordering::Relaxed);

                tracing::debug!(
                    model = %model,
                    pending = entry.requests.len(),
                    "Request added to batch"
                );

                // Flush if batch size reached
                if entry.requests.len() >= batch_size {
                    flush_batch(model, &mut accumulator, &stats).await;
                }
            }
            // Timeout flush
            _ = tokio::time::sleep(batch_timeout) => {
                flush_ready_batches(&mut accumulator, &stats, batch_timeout).await;
            }
        }
    }

    let batch_count = stats.batch_count.load(Ordering::Relaxed);
    let total_requests = stats.total_requests.load(Ordering::Relaxed);
    tracing::info!(
        batch_count,
        total_requests,
        avg_batch_size = if batch_count > 0 { total_requests as f64 / batch_count as f64 } else { 0.0 },
        "Batch processor shutting down"
    );
}

async fn flush_batch(
    state: &Arc<AppState>,
    model: String,
    accumulator: &mut HashMap<String, BatchAccumulator>,
    batch_count: &mut u64,
) {
    if let Some(entry) = accumulator.remove(&model) {
        if entry.requests.is_empty() {
            return;
        }

        *batch_count += 1;

        tracing::info!(
            model = %model,
            batch_size = entry.requests.len(),
            "Executing batch"
        );

        // Process all requests in parallel
        let engine = state.engine.read().await;
        let tasks: Vec<_> = entry
            .requests
            .into_iter()
            .map(|item| {
                let engine_clone = &engine;
                async move {
                    let prompt = messages_to_prompt(&item.req.messages);
                    let max_tokens = item.req.max_tokens.unwrap_or(512) as usize;

                    let text = engine_clone.generate(&prompt, max_tokens).await;

                    let response = ChatCompletionsResponse {
                        id: format!("chatcmpl-{}", uuid_simple()),
                        object: "chat.completion".into(),
                        created: unix_timestamp(),
                        model: item.req.model.clone(),
                        choices: vec![Choice {
                            index: 0,
                            message: Message {
                                role: "assistant".into(),
                                content: text,
                            },
                            finish_reason: "stop".into(),
                        }],
                        usage: Usage {
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            total_tokens: 0,
                        },
                    };

                    let _ = item.tx.send(response);
                }
            })
            .collect();

        // Execute all tasks concurrently
        futures::future::join_all(tasks).await;
    }
}

async fn flush_ready_batches(
    state: &Arc<AppState>,
    accumulator: &mut HashMap<String, BatchAccumulator>,
    batch_count: &mut u64,
    timeout: Duration,
) {
    let now = Instant::now();
    let ready_models: Vec<_> = accumulator
        .iter()
        .filter(|(_, entry)| now.duration_since(entry.last_add) >= timeout)
        .map(|(model, _)| model.clone())
        .collect();

    for model in ready_models {
        flush_batch(state, model, accumulator, batch_count).await;
    }
}

async fn flush_all_batches(state: &Arc<AppState>, accumulator: &mut HashMap<String, BatchAccumulator>) {
    let models: Vec<_> = accumulator.keys().cloned().collect();
    let mut batch_count = 0u64;
    for model in models {
        flush_batch(state, model, accumulator, &mut batch_count).await;
    }
}

/// Batch statistics endpoint
pub async fn batch_stats(
    State(_state): State<Arc<AppState>>,
) -> (StatusCode, Json<BatchStats>) {
    // In a real implementation, these would be tracked atomically
    let stats = BatchStats {
        batch_count: 0,
        total_requests: 0,
        avg_batch_size: 0.0,
    };

    (StatusCode::OK, Json(stats))
}

fn messages_to_prompt(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{:x}{:x}", dur.as_secs(), dur.subsec_nanos())
}

fn unix_timestamp() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
