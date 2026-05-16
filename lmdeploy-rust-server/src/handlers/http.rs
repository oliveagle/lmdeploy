use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, Sse},
    Json,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

use crate::cache::compute_hash;
use crate::metrics::StreamMetricsSnapshot;
use crate::server::{AppState, BatchItem, BatchStatsResponse};

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub user: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Clone)]
pub struct Choice {
    pub index: i32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<DeltaChoice>,
}

#[derive(Debug, Serialize)]
pub struct DeltaChoice {
    pub index: i32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    pub content: Option<String>,
    pub role: Option<String>,
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> (StatusCode, Json<ChatCompletionsResponse>) {
    let model = req.model.clone();
    let start = std::time::Instant::now();

    let prompt = messages_to_prompt(&req.messages);

    tracing::info!(model = %model, prompt_len = prompt.len(), "Chat completions request");

    // Use batch processor if enabled and not streaming
    let response = if let Some(batch_tx) = &state.batch_sender {
        // Batch mode: send to batch processor and wait for response
        let (tx, rx) = oneshot::channel();
        let batch_item = BatchItem {
            req: req.clone(),
            tx,
        };

        if batch_tx.send(batch_item).is_ok() {
            match rx.await {
                Ok(resp) => resp,
                Err(_) => ChatCompletionsResponse {
                    id: format!("chatcmpl-{}", uuid_simple()),
                    object: "chat.completion".into(),
                    created: unix_timestamp(),
                    model,
                    choices: vec![Choice {
                        index: 0,
                        message: Message {
                            role: "assistant".into(),
                            content: "Batch processor error".into(),
                        },
                        finish_reason: "error".into(),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                },
            }
        } else {
            // Batch channel closed, fallback to direct mode
            fallback_chat_completion(&state, &req, &model).await
        }
    } else {
        // Direct mode: process immediately
        fallback_chat_completion(&state, &req, &model).await
    };

    tracing::info!(latency_ms = start.elapsed().as_millis(), "Chat completions done");
    (StatusCode::OK, Json(response))
}

async fn fallback_chat_completion(
    state: &Arc<AppState>,
    req: &ChatCompletionsRequest,
    model: &str,
) -> ChatCompletionsResponse {
    let prompt = messages_to_prompt(&req.messages);
    let engine = state.engine.read().await;
    let text = engine.generate(&prompt, req.max_tokens.unwrap_or(512) as usize).await;

    ChatCompletionsResponse {
        id: format!("chatcmpl-{}", uuid_simple()),
        object: "chat.completion".into(),
        created: unix_timestamp(),
        model: model.to_string(),
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
    }
}

pub async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let model = req.model.clone();
    let id = format!("chatcmpl-{}", uuid_simple());
    let created = unix_timestamp();
    let config = state.config.read().await;
    let stream_timeout_ms = config.server.stream_keepalive_interval_ms;
    drop(config);
    let metrics = state.metrics.clone();

    tracing::info!(model = %model, "Chat completions stream request");

    metrics.streams.record_stream_start(0);

    let prompt = messages_to_prompt(&req.messages);
    let engine = state.engine.read().await;
    let chunks = engine.generate_stream(&prompt).await;

    let stream_id = id.clone();
    let stream_model = model.clone();
    let final_id = id.clone();
    let final_model = model.clone();
    let first_token_start = Instant::now();
    let metrics_inner = metrics.clone();

    let stream = chunks.map(move |chunk_text| {
        // Record first token latency on first chunk
        if first_token_start.elapsed().as_millis() > 0 {
            metrics_inner.streams.record_stream_start(first_token_start.elapsed().as_micros() as u64);
        }
        metrics_inner.streams.record_chunk();

        let chunk = ChatCompletionChunk {
            id: stream_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: stream_model.clone(),
            choices: vec![DeltaChoice {
                index: 0,
                delta: Delta {
                    content: Some(chunk_text),
                    role: None,
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default()
            .event("chat.completion.chunk")
            .data(json))
    }).chain(futures::stream::once(async move {
        let final_chunk = ChatCompletionChunk {
            id: final_id,
            object: "chat.completion.chunk".into(),
            created,
            model: final_model,
            choices: vec![DeltaChoice {
                index: 0,
                delta: Delta { content: None, role: None },
                finish_reason: Some("stop".into()),
            }],
        };

        let json = serde_json::to_string(&final_chunk).unwrap_or_default();
        Ok(Event::default()
            .event("chat.completion.chunk")
            .data(json))
    }));

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_millis(stream_timeout_ms)))
}

#[derive(Debug, Deserialize)]
pub struct CompletionsRequest {
    pub model: String,
    pub prompt: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub stream: Option<bool>,
    pub echo: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: i32,
    pub finish_reason: String,
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionsRequest>,
) -> (StatusCode, Json<CompletionsResponse>) {
    let model = req.model.clone();
    let start = std::time::Instant::now();

    let response = if let Some(batch_tx) = &state.batch_sender {
        // Batch mode: send to batch processor and wait for response
        let (tx, rx) = oneshot::channel();
        let batch_item = BatchItem {
            req: ChatCompletionsRequest {
                model: req.model.clone(),
                messages: vec![Message {
                    role: "user".into(),
                    content: req.prompt.clone(),
                }],
                temperature: req.temperature,
                top_p: None,
                max_tokens: req.max_tokens,
                stream: req.stream,
                stop: None,
                seed: None,
                presence_penalty: None,
                frequency_penalty: None,
                user: None,
            },
            tx,
        };

        if batch_tx.send(batch_item).is_ok() {
            match rx.await {
                Ok(chat_resp) => {
                    // Convert ChatCompletionsResponse to CompletionsResponse
                    CompletionsResponse {
                        id: chat_resp.id,
                        object: "text_completion".into(),
                        created: chat_resp.created,
                        model: chat_resp.model,
                        choices: chat_resp.choices.iter().map(|c| CompletionChoice {
                            text: c.message.content.clone(),
                            index: c.index,
                            finish_reason: c.finish_reason.clone(),
                        }).collect(),
                        usage: chat_resp.usage,
                    }
                }
                Err(_) => CompletionsResponse {
                    id: format!("cmpl-{}", uuid_simple()),
                    object: "text_completion".into(),
                    created: unix_timestamp(),
                    model,
                    choices: vec![CompletionChoice {
                        text: "Batch processor error".into(),
                        index: 0,
                        finish_reason: "error".into(),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                },
            }
        } else {
            // Batch channel closed, fallback to direct mode
            fallback_completions(&state, &req, &model).await
        }
    } else {
        // Direct mode: process immediately
        fallback_completions(&state, &req, &model).await
    };

    tracing::info!(latency_ms = start.elapsed().as_millis(), "Completions done");
    (StatusCode::OK, Json(response))
}

async fn fallback_completions(
    state: &Arc<AppState>,
    req: &CompletionsRequest,
    model: &str,
) -> CompletionsResponse {
    let engine = state.engine.read().await;
    let text = engine.generate(&req.prompt, req.max_tokens.unwrap_or(512) as usize).await;

    CompletionsResponse {
        id: format!("cmpl-{}", uuid_simple()),
        object: "text_completion".into(),
        created: unix_timestamp(),
        model: model.to_string(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

pub async fn list_models() -> (StatusCode, Json<ModelsResponse>) {
    (StatusCode::OK, Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelInfo {
            id: "default-model".into(),
            object: "model".into(),
            created: unix_timestamp(),
            owned_by: "lmdeploy".into(),
        }],
    }))
}

/// Batch chat completions request
#[derive(Debug, Deserialize, Clone)]
pub struct BatchChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Vec<Message>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i32>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub user: Option<String>,
}

/// Batch chat completions response
#[derive(Debug, Serialize)]
pub struct BatchChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<BatchChoice>,
    pub usage: BatchUsage,
}

#[derive(Debug, Serialize)]
pub struct BatchChoice {
    pub index: i32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct BatchUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// Batch completions request
#[derive(Debug, Deserialize, Clone)]
pub struct BatchCompletionsRequest {
    pub model: String,
    pub prompts: Vec<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub echo: Option<bool>,
}

/// Batch completions response
#[derive(Debug, Serialize)]
pub struct BatchCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<BatchCompletionChoice>,
    pub usage: BatchUsage,
}

#[derive(Debug, Serialize)]
pub struct BatchCompletionChoice {
    pub text: String,
    pub index: i32,
    pub finish_reason: String,
}

/// Batch chat completions endpoint (OpenAI-compatible batch format)
pub async fn batch_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchChatCompletionsRequest>,
) -> (StatusCode, Json<BatchChatCompletionsResponse>) {
    let model = req.model.clone();
    let start = std::time::Instant::now();

    tracing::info!(
        model = %model,
        batch_size = req.messages.len(),
        "Batch chat completions request"
    );

    let engine = state.engine.read().await;
    let mut choices = Vec::new();
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;

    for (idx, messages) in req.messages.iter().enumerate() {
        let prompt = messages_to_prompt(messages);
        let max_tokens = req.max_tokens.unwrap_or(512) as usize;
        let text = engine.generate(&prompt, max_tokens).await;

        total_prompt_tokens += prompt.len() as i32;
        total_completion_tokens += text.len() as i32;

        choices.push(BatchChoice {
            index: idx as i32,
            message: Message {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: "stop".into(),
        });
    }

    let response = BatchChatCompletionsResponse {
        id: format!("chatcmpl-{}", uuid_simple()),
        object: "chat.completion".into(),
        created: unix_timestamp(),
        model,
        choices,
        usage: BatchUsage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
        },
    };

    tracing::info!(latency_ms = start.elapsed().as_millis(), "Batch chat completions done");
    (StatusCode::OK, Json(response))
}

/// Batch completions endpoint
pub async fn batch_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchCompletionsRequest>,
) -> (StatusCode, Json<BatchCompletionsResponse>) {
    let model = req.model.clone();
    let start = std::time::Instant::now();

    tracing::info!(
        model = %model,
        batch_size = req.prompts.len(),
        "Batch completions request"
    );

    let engine = state.engine.read().await;
    let mut choices = Vec::new();
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;

    for (idx, prompt) in req.prompts.iter().enumerate() {
        let max_tokens = req.max_tokens.unwrap_or(512) as usize;
        let text = engine.generate(prompt, max_tokens).await;

        total_prompt_tokens += prompt.len() as i32;
        total_completion_tokens += text.len() as i32;

        choices.push(BatchCompletionChoice {
            text,
            index: idx as i32,
            finish_reason: "stop".into(),
        });
    }

    let response = BatchCompletionsResponse {
        id: format!("cmpl-{}", uuid_simple()),
        object: "text_completion".into(),
        created: unix_timestamp(),
        model,
        choices,
        usage: BatchUsage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
        },
    };

    tracing::info!(latency_ms = start.elapsed().as_millis(), "Batch completions done");
    (StatusCode::OK, Json(response))
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

pub async fn health_check() -> (StatusCode, Json<HealthResponse>) {
    (StatusCode::OK, Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    }))
}

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub token_ids: Vec<u32>,
    pub length: usize,
    pub hash: String,
    pub cached: bool,
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> (StatusCode, Json<TokenizeResponse>) {
    let hash = compute_hash(&req.text);
    let start = std::time::Instant::now();

    let token_ids = state
        .tokenizer_cache
        .get_or_tokenize(&req.text, |text| {
            let owned = text.to_string();
            async move {
                let mock_tokens: Vec<u32> = owned.chars().map(|c| c as u32).collect();
                Ok(mock_tokens)
            }
        })
        .await
        .unwrap_or_else(|_| req.text.chars().map(|c| c as u32).collect());

    let cached = start.elapsed().as_millis() < 1; // Fast response = cache hit

    tracing::info!(
        text_len = req.text.len(),
        token_count = token_ids.len(),
        cached,
        latency_ms = start.elapsed().as_millis(),
        "Tokenize request"
    );

    let length = token_ids.len();

    (
        StatusCode::OK,
        Json(TokenizeResponse {
            token_ids,
            length,
            hash,
            cached,
        }),
    )
}

#[derive(Debug, Serialize)]
pub struct CacheMetricsResponse {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefix_hits: u64,
    pub evictions: u64,
    pub hit_rate: f64,
    pub current_size: usize,
}

pub async fn cache_metrics(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<CacheMetricsResponse>) {
    let metrics = state.tokenizer_cache.metrics().await;
    let hit_rate = state.tokenizer_cache.hit_rate().await;
    let size = state.tokenizer_cache.size().await;

    (
        StatusCode::OK,
        Json(CacheMetricsResponse {
            total_requests: metrics.total_requests,
            cache_hits: metrics.cache_hits,
            cache_misses: metrics.cache_misses,
            prefix_hits: metrics.prefix_hits,
            evictions: metrics.evictions,
            hit_rate,
            current_size: size,
        }),
    )
}

#[derive(Debug, Serialize)]
pub struct ClearCacheResponse {
    pub status: String,
    pub message: String,
}

pub async fn clear_cache(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ClearCacheResponse>) {
    state.tokenizer_cache.clear().await;

    tracing::info!("Cache cleared via API");

    (
        StatusCode::OK,
        Json(ClearCacheResponse {
            status: "ok".into(),
            message: "Cache cleared successfully".into(),
        }),
    )
}

/// Batch statistics endpoint
pub async fn batch_stats(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<BatchStatsResponse>) {
    let stats = state.batch_stats.batch_response();
    tracing::debug!(
        batch_count = stats.batch_count,
        total_requests = stats.total_requests,
        avg_batch_size = stats.avg_batch_size,
        "Batch stats requested"
    );
    (StatusCode::OK, Json(stats))
}

/// Stream metrics endpoint
pub async fn stream_metrics(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<StreamMetricsSnapshot>) {
    let metrics = state.metrics.streams.snapshot();
    tracing::debug!(
        total_streams = metrics.total_streams,
        avg_first_token_latency_ms = metrics.avg_first_token_latency_ms,
        "Stream metrics requested"
    );
    (StatusCode::OK, Json(metrics))
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
