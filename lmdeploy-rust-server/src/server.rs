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
use crate::handlers::http::{
    batch_chat_completions, batch_completions, batch_stats, cache_metrics, chat_completions,
    chat_completions_stream, clear_cache, completions, embeddings, get_prefix_cache_status,
    health_check, list_models, model_load, model_load_progress, model_reload, model_unload,
    rate_limit_status, set_prefix_cache, stream_metrics, tokenize,
    ChatCompletionsRequest, ChatCompletionsResponse, Choice, ChoiceLogprobs, Message,
    response_format_to_grammar, TopLogprobEntry, Usage,
};
use crate::metrics::{init_metrics, AppMetrics};
use crate::model::{ModelLoadTracker, ModelManager};
use crate::rate_limiter::{GlobalRateLimiter, PerIpRateLimiter};

use crate::metrics::increment_tokens_generated_total;
use crate::model::GenerationParams;

/// Type alias for the batch sender channel
pub type BatchSender = mpsc::UnboundedSender<BatchItem>;

#[derive(Clone)]
pub struct AppState {
    /// Main model manager for routing
    pub model_manager: Arc<RwLock<ModelManager>>,
    /// Tokenizer cache
    pub tokenizer_cache: Arc<TokenizeCache>,
    /// Application config
    pub config: Arc<RwLock<AppConfig>>,
    /// Request semaphore for rate limiting
    pub request_semaphore: Arc<Semaphore>,
    /// Batch sender channel
    pub batch_sender: Option<Arc<BatchSender>>,
    /// Batch statistics
    pub batch_stats: Arc<BatchStats>,
    /// Application metrics
    pub metrics: Arc<AppMetrics>,
    /// Config reload sender
    pub config_reload_tx: mpsc::Sender<()>,
    /// Model loading progress tracker
    pub model_load_tracker: Arc<ModelLoadTracker>,
    /// Global rate limiter
    pub global_rate_limiter: Option<Arc<GlobalRateLimiter>>,
    /// Per-IP rate limiter
    pub per_ip_rate_limiter: Option<Arc<PerIpRateLimiter>>,
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

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            batch_count: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
        }
    }
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
        Self::default()
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
    // Parse engine type from config
    let engine_type = crate::model::cpp_engine::EngineType::from_str(&config.model.engine_type)
        .unwrap_or(crate::model::cpp_engine::EngineType::PureCpp);

    // Read prefix caching config
    let prefix_cache_enabled = config.model.prefix_cache_enabled;

    // Initialize ModelManager with default model using specified engine type
    let model_manager = Arc::new(RwLock::new(
        ModelManager::with_default_model_and_type_and_prefix_cache(
            &config.model.model_path,
            engine_type,
            prefix_cache_enabled,
        )
        .await?,
    ));
    let grpc_model_manager = model_manager.clone();

    // Log default model info
    {
        let mm = model_manager.read().await;
        tracing::info!(
            default_model = %mm.default_model(),
            model_count = mm.model_count(),
            "Model manager initialized"
        );

        // Load additional models from config if multi_model is enabled
        if config.multi_model.enabled {
            for entry in &config.multi_model.models {
                tracing::info!(
                    model_name = %entry.name,
                    model_path = %entry.path,
                    engine_type = %entry.engine_type,
                    prefix_cache_enabled = entry.prefix_cache_enabled.unwrap_or(config.model.prefix_cache_enabled),
                    "Loading additional model from config"
                );
                let entry_engine_type =
                    crate::model::cpp_engine::EngineType::from_str(&entry.engine_type)
                        .unwrap_or(crate::model::cpp_engine::EngineType::PureCpp);
                let entry_prefix_cache_enabled =
                    entry.prefix_cache_enabled.unwrap_or(config.model.prefix_cache_enabled);
                let mut mm = model_manager.write().await;
                if let Err(e) = mm
                    .load_model_with_type_and_prefix_cache(
                        &entry.name,
                        &entry.path,
                        entry_engine_type,
                        entry_prefix_cache_enabled,
                    )
                    .await
                {
                    tracing::warn!(
                        error = %e,
                        model_name = %entry.name,
                        "Failed to load additional model, skipping"
                    );
                }
            }
        }
    }

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
    let metrics = Arc::new(AppMetrics::new());

    // Initialize rate limiters
    let (global_rate_limiter, per_ip_rate_limiter) = if config.server.rate_limit.enabled {
        let global = Arc::new(GlobalRateLimiter::new(
            config.server.rate_limit.requests_per_second,
            config.server.rate_limit.burst_size,
        ));
        let per_ip = Arc::new(PerIpRateLimiter::new(
            config.server.rate_limit.per_ip.requests_per_second,
            config.server.rate_limit.per_ip.burst_size,
            config.server.rate_limit.per_ip.max_tracked_ips,
        ));
        tracing::info!(
            rate_limit_enabled = true,
            global_rps = config.server.rate_limit.requests_per_second,
            global_burst = config.server.rate_limit.burst_size,
            per_ip_enabled = config.server.rate_limit.per_ip.enabled,
            per_ip_rps = config.server.rate_limit.per_ip.requests_per_second,
            "Rate limiters initialized"
        );
        (
            Some(global),
            if config.server.rate_limit.per_ip.enabled {
                Some(per_ip)
            } else {
                None
            },
        )
    } else {
        tracing::info!("Rate limiting disabled");
        (None, None)
    };

    // Initialize Prometheus metrics exporter
    init_metrics(&config.metrics);
    tracing::info!(
        metrics_enabled = config.metrics.enabled,
        metrics_host = config.metrics.host,
        metrics_port = config.metrics.port,
        "Metrics system initialized"
    );

    let http_addr = format!("{}:{}", config.server.http_addr, config.server.http_port)
        .parse::<SocketAddr>()
        .map_err(|e| AppError::Config(format!("Invalid HTTP address: {e}")))?;
    let grpc_addr = format!("{}:{}", config.server.grpc_addr, config.server.grpc_port)
        .parse::<SocketAddr>()
        .map_err(|e| AppError::Config(format!("Invalid gRPC address: {e}")))?;

    // Spawn batch processor if enabled
    let batch_processor_handle = if config.server.batch_enabled {
        let (batch_tx, batch_rx) = mpsc::unbounded_channel::<BatchItem>();
        let batch_size = config.server.batch_size;
        let batch_timeout_ms = config.server.batch_timeout_ms;
        let batch_timeout = Duration::from_millis(config.server.batch_timeout_ms);
        let stats_clone = batch_stats.clone();
        let manager_clone = model_manager.clone();

        let handle = tokio::spawn(async move {
            tracing::info!(
                batch_size,
                timeout_ms = batch_timeout_ms,
                "Batch processor started"
            );
            run_batch_processor(
                batch_rx,
                batch_size,
                batch_timeout,
                stats_clone,
                manager_clone,
            )
            .await
        });

        tracing::info!(
            batch_size = config.server.batch_size,
            timeout_ms = config.server.batch_timeout_ms,
            "Batch processor started"
        );
        Some((Arc::new(batch_tx), handle))
    } else {
        None
    };

    let (config_reload_tx, mut config_reload_rx) = mpsc::channel::<()>(32);

    let model_load_tracker = Arc::new(ModelLoadTracker::new());

    let state = Arc::new(AppState {
        model_manager,
        tokenizer_cache: tokenizer_cache.clone(),
        config: Arc::new(RwLock::new(config.clone())),
        request_semaphore,
        batch_sender: batch_processor_handle.as_ref().map(|(tx, _)| tx.clone()),
        batch_stats,
        metrics: metrics.clone(),
        config_reload_tx: config_reload_tx.clone(),
        model_load_tracker,
        global_rate_limiter,
        per_ip_rate_limiter,
    });

    // Spawn config reload task
    let config_state = state.clone();
    tokio::spawn(async move {
        while config_reload_rx.recv().await.is_some() {
            match AppConfig::load() {
                Ok(new_config) => {
                    let mut config = config_state.config.write().await;
                    *config = new_config.clone();
                    tracing::info!(
                        config = ?new_config,
                        "Configuration reloaded successfully"
                    );

                    // Update log level if changed
                    // Note: Can't re-init subscriber, just log the change
                    tracing::debug!("Log level changed to: {}", new_config.logging.level);
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to reload configuration");
                }
            }
        }
    });

    // Spawn gRPC server
    let grpc_handle = tokio::spawn(async move {
        crate::grpc::start_grpc_server(
            grpc_addr,
            env!("CARGO_PKG_VERSION").to_string(),
            tokenizer_cache,
            grpc_model_manager,
        )
        .await
    });

    // Spawn SIGHUP handler for hot reload
    #[cfg(unix)]
    {
        let reload_tx = config_reload_tx.clone();
        tokio::spawn(async move {
            let mut sighup_stream = signal::unix::signal(signal::unix::SignalKind::hangup())
                .expect("failed to install SIGHUP handler");
            while sighup_stream.recv().await.is_some() {
                tracing::info!("SIGHUP received, triggering configuration reload");
                if reload_tx.send(()).await.is_err() {
                    tracing::warn!("Failed to send config reload signal on SIGHUP");
                }
            }
        });
    }

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
    if let Some((_, handle)) = batch_processor_handle {
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
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/chat/completions/batch", post(batch_chat_completions))
        .route("/v1/completions/batch", post(batch_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions/stream", post(chat_completions_stream))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/cache/metrics", get(cache_metrics))
        .route("/v1/cache/clear", post(clear_cache))
        .route("/v1/cache/prefix-cache", get(get_prefix_cache_status))
        .route("/v1/cache/prefix-cache", post(set_prefix_cache))
        .route("/v1/metrics/stream", get(stream_metrics))
        .route("/v1/batch/stats", get(batch_stats))
        .route("/v1/config/reload", post(reload_config))
        .route("/v1/config", get(get_config))
        .route("/v1/models/load", post(model_load))
        .route("/v1/models/unload", post(model_unload))
        .route("/v1/models/reload", post(model_reload))
        .route("/v1/models/progress", post(model_load_progress))
        .route("/v1/rate-limit/status", get(rate_limit_status))
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

/// Trigger configuration reload (called by POST /v1/config/reload)
pub async fn reload_config(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    tracing::info!("Configuration reload requested");
    match state.config_reload_tx.send(()).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "config_reload_triggered"
            })),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "status": "config_reload_failed"
            })),
        ),
    }
}

/// Get current configuration (called by GET /v1/config)
pub async fn get_config(State(state): State<Arc<AppState>>) -> Json<AppConfig> {
    let config = state.config.read().await.clone();
    Json(config)
}

/// Batch processor for accumulating and executing inference requests
async fn run_batch_processor(
    mut rx: mpsc::UnboundedReceiver<BatchItem>,
    batch_size: usize,
    batch_timeout: Duration,
    stats: Arc<BatchStats>,
    manager: Arc<RwLock<ModelManager>>,
) {
    let mut accumulator: HashMap<String, BatchAccumulator> = HashMap::new();

    loop {
        tokio::select! {
            // Receive new request
            item_opt = rx.recv() => {
                let Some(item) = item_opt else {
                    // Channel closed, flush remaining batches
                    flush_all_batches(&mut accumulator, &stats, &manager).await;
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
                    flush_batch(model, &mut accumulator, &stats, manager.clone()).await;
                }
            }
            // Timeout flush
            _ = tokio::time::sleep(batch_timeout) => {
                flush_ready_batches(&mut accumulator, &stats, batch_timeout, &manager).await;
            }
        }
    }

    let batch_count = stats.batch_count.load(Ordering::Relaxed);
    let total_requests = stats.total_requests.load(Ordering::Relaxed);
    tracing::info!(
        batch_count,
        total_requests,
        avg_batch_size = if batch_count > 0 {
            total_requests as f64 / batch_count as f64
        } else {
            0.0
        },
        "Batch processor shutting down"
    );
}

async fn flush_batch(
    model: String,
    accumulator: &mut HashMap<String, BatchAccumulator>,
    stats: &Arc<BatchStats>,
    manager: Arc<RwLock<ModelManager>>,
) {
    if let Some(entry) = accumulator.remove(&model) {
        if entry.requests.is_empty() {
            return;
        }

        stats.batch_count.fetch_add(1, Ordering::Relaxed);
        let batch_size = entry.requests.len();

        tracing::info!(
            model = %model,
            batch_size,
            "Executing batch"
        );

        // Get the engine for this model
        let mm = manager.read().await;
        let engine = match mm.get_model(Some(&model)) {
            Some(e) => e,
            None => {
                tracing::error!(model = %model, "Model not found for batch execution");
                return;
            }
        };
        drop(mm);

        // Process all requests in parallel
        let tasks: Vec<_> = entry
            .requests
            .into_iter()
            .map(|item| {
                let engine_clone = engine.clone();
                async move {
                    let prompt = messages_to_prompt(&item.req.messages);

                    let need_logprobs = item.req.logprobs.unwrap_or(false)
                        || item.req.top_logprobs.unwrap_or(0) > 0;

                    let grammar = response_format_to_grammar(&item.req.response_format);
                    let params = GenerationParams::from_chat_request(
                        item.req.temperature,
                        item.req.top_p,
                        item.req.top_k,
                        item.req.min_p,
                        item.req.max_tokens,
                        item.req.seed,
                        item.req.presence_penalty,
                        item.req.frequency_penalty,
                        item.req.stop.clone(),
                        item.req.logprobs,
                        item.req.top_logprobs,
                        grammar,
                    );

                    // Call the actual engine
                    let eng = engine_clone.read().await;
                    let (text, logprobs) = if need_logprobs {
                        let (t, _nt, _el, lp) = eng.generate_with_logprobs(&prompt, params).await;
                        (t, lp)
                    } else {
                        let t = eng.generate(&prompt, params).await;
                        (t, None)
                    };

                    let response = ChatCompletionsResponse {
                        id: format!("chatcmpl-{}", uuid_simple()),
                        object: "chat.completion".into(),
                        created: unix_timestamp(),
                        model: item.req.model.clone(),
                        choices: vec![Choice {
                            index: 0,
                            message: Message {
                                role: "assistant".into(),
                                content: text.clone(),
                            },
                            finish_reason: "stop".into(),
                            logprobs: logprobs.map(|lp| ChoiceLogprobs {
                                tokens: lp.iter().map(|t| t.token.clone()).collect(),
                                token_logprobs: lp.iter().map(|t| t.logprob).collect(),
                                top_logprobs: lp
                                    .iter()
                                    .map(|t| {
                                        if !t.top_logprobs.is_empty() {
                                            let first = t.top_logprobs.first().unwrap();
                                            Some(TopLogprobEntry {
                                                token: first.token.clone(),
                                                logprob: first.logprob,
                                                bytes: first.bytes.clone(),
                                            })
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                                top_tokens: Vec::new(),
                            }),
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
    accumulator: &mut HashMap<String, BatchAccumulator>,
    stats: &Arc<BatchStats>,
    timeout: Duration,
    manager: &Arc<RwLock<ModelManager>>,
) {
    let now = Instant::now();
    let ready_models: Vec<_> = accumulator
        .iter()
        .filter(|(_, entry)| now.duration_since(entry.last_add) >= timeout)
        .map(|(model, _)| model.clone())
        .collect();

    for model in ready_models {
        flush_batch(model, accumulator, stats, manager.clone()).await;
    }
}

async fn flush_all_batches(
    accumulator: &mut HashMap<String, BatchAccumulator>,
    stats: &Arc<BatchStats>,
    manager: &Arc<RwLock<ModelManager>>,
) {
    let models: Vec<_> = accumulator.keys().cloned().collect();
    for model in models {
        flush_batch(model, accumulator, stats, manager.clone()).await;
    }
}

fn messages_to_prompt(messages: &[Message]) -> String {
    // Use ChatML format which is compatible with Qwen, LLaMA-2, etc.
    // Format: <|im_start|>{role}\n{content}<|im_end|>
    let mut prompt = String::new();
    for m in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&m.role);
        prompt.push('\n');
        prompt.push_str(&m.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
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
