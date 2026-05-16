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
    pub stop: Option<Stop>,
    pub seed: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub n: Option<u32>,
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub user: Option<String>,
}

/// Stop sequence(s) - can be a string or array of strings
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum Stop {
    Single(String),
    Multiple(Vec<String>),
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

    // Get the appropriate engine for the requested model
    let mm = state.model_manager.read().await;
    let engine = match mm.get_model(Some(model)) {
        Some(e) => e,
        None => {
            tracing::warn!(model = %model, "Model not found, using default");
            mm.get_model(None).unwrap()
        }
    };
    drop(mm);

    let eng = engine.read().await;
    let text = eng.generate(&prompt, req.max_tokens.unwrap_or(512) as usize).await;

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
    let _stream_timeout_ms = config.server.stream_timeout_secs * 1000;
    let keepalive_interval_ms = config.server.stream_keepalive_interval_ms;
    drop(config);
    let metrics = state.metrics.clone();

    tracing::info!(model = %model, "Chat completions stream request");

    let prompt = messages_to_prompt(&req.messages);

    // Get the appropriate engine for the requested model
    let mm = state.model_manager.read().await;
    let engine = match mm.get_model(Some(&model)) {
        Some(e) => e,
        None => {
            tracing::warn!(model = %model, "Model not found, using default");
            mm.get_model(None).unwrap()
        }
    };
    drop(mm);

    let chunks = engine.read().await.generate_stream(&prompt).await;

    let stream_id = id.clone();
    let stream_model = model.clone();
    let final_id = id.clone();
    let final_model = model.clone();
    let first_token_start = Instant::now();
    let metrics_inner = metrics.clone();

    // Create stream with timeout handling
    let stream = chunks.map(move |chunk_text| {
        // Record first token latency on first chunk (in milliseconds)
        let latency_ms = first_token_start.elapsed().as_millis() as u64;
        metrics_inner.streams.record_stream_start(latency_ms);

        tracing::debug!(
            first_token_latency_ms = latency_ms,
            "First token latency"
        );

        // Log warning if latency exceeds threshold
        if latency_ms > 50 {
            tracing::warn!(
                first_token_latency_ms = latency_ms,
                threshold_ms = 50,
                "First token latency exceeds threshold"
            );
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

    // Apply keep-alive and return SSE response
    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_millis(keepalive_interval_ms))
        )
}

#[derive(Debug, Deserialize)]
pub struct CompletionsRequest {
    pub model: String,
    pub prompt: Prompt,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub stream: Option<bool>,
    pub echo: Option<bool>,
    pub suffix: Option<String>,
    pub stop: Option<Stop>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub n: Option<u32>,
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    pub best_of: Option<u32>,
    pub user: Option<String>,
}

/// Prompt - can be a string or array of strings
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum Prompt {
    Single(String),
    Multiple(Vec<String>),
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

    // Convert Prompt enum to string
    let prompt_text = match &req.prompt {
        Prompt::Single(s) => s.clone(),
        Prompt::Multiple(vec) => vec.join("\n"),
    };

    let response = if let Some(batch_tx) = &state.batch_sender {
        // Batch mode: send to batch processor and wait for response
        let (tx, rx) = oneshot::channel();
        let batch_item = BatchItem {
            req: ChatCompletionsRequest {
                model: req.model.clone(),
                messages: vec![Message {
                    role: "user".into(),
                    content: prompt_text.clone(),
                }],
                temperature: req.temperature,
                top_p: None,
                max_tokens: req.max_tokens,
                stream: req.stream,
                stop: req.stop.clone(),
                seed: None,
                presence_penalty: req.presence_penalty,
                frequency_penalty: req.frequency_penalty,
                n: req.n,
                logit_bias: req.logit_bias.clone(),
                logprobs: None,
                top_logprobs: None,
                user: req.user.clone(),
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
    // Convert Prompt enum to string
    let prompt_text = match &req.prompt {
        Prompt::Single(s) => s.as_str(),
        Prompt::Multiple(vec) => {
            let joined = vec.join("\n");
            Box::leak(joined.into_boxed_str())
        }
    };

    let mm = state.model_manager.read().await;
    let engine = match mm.get_model(Some(model)) {
        Some(e) => e,
        None => mm.get_model(None).unwrap(),
    };
    drop(mm);

    let eng = engine.read().await;
    let text = eng.generate(prompt_text, req.max_tokens.unwrap_or(512) as usize).await;

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

pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ModelsResponse>) {
    let models = state.model_manager.read().await.list_models().await;

    let data: Vec<ModelInfo> = models
        .iter()
        .map(|m| ModelInfo {
            id: m.name.clone(),
            object: "model".into(),
            created: m.loaded_at.unwrap_or(unix_timestamp()),
            owned_by: "lmdeploy".into(),
        })
        .collect();

    (StatusCode::OK, Json(ModelsResponse {
        object: "list".into(),
        data,
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

    // Get the appropriate engine for the requested model
    let mm = state.model_manager.read().await;
    let engine = match mm.get_model(Some(&model)) {
        Some(e) => e,
        None => mm.get_model(None).unwrap(),
    };
    drop(mm);

    let eng = engine.read().await;
    let mut choices = Vec::new();
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;

    for (idx, messages) in req.messages.iter().enumerate() {
        let prompt = messages_to_prompt(messages);
        let max_tokens = req.max_tokens.unwrap_or(512) as usize;
        let text = eng.generate(&prompt, max_tokens).await;

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

    // Get the appropriate engine for the requested model
    let mm = state.model_manager.read().await;
    let engine = match mm.get_model(Some(&model)) {
        Some(e) => e,
        None => mm.get_model(None).unwrap(),
    };
    drop(mm);

    let eng = engine.read().await;
    let mut choices = Vec::new();
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;

    for (idx, prompt) in req.prompts.iter().enumerate() {
        let max_tokens = req.max_tokens.unwrap_or(512) as usize;
        let text = eng.generate(prompt, max_tokens).await;

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

// ============================================================================
// Embeddings API (OpenAI-compatible)
// ============================================================================

/// Embeddings request - matches OpenAI /v1/embeddings API
#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingsRequest {
    pub model: Option<String>,
    /// Input text to embed. Can be a string or array of strings.
    pub input: EmbeddingInput,
    pub encoding_format: Option<String>,
    pub dimensions: Option<u32>,
    pub user: Option<String>,
}

/// Embedding input - can be a single string or array of strings
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Embeddings response - matches OpenAI /v1/embeddings API
#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Single embedding data
#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Embedding usage statistics
#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

/// Embeddings endpoint (OpenAI-compatible)
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> (StatusCode, Json<EmbeddingsResponse>) {
    let start = std::time::Instant::now();
    let model = req.model.clone().unwrap_or_else(|| "default-embedding-model".to_string());

    tracing::info!(
        model = %model,
        "Embeddings request"
    );

    // Process inputs (single or multiple)
    let inputs: Vec<String> = match &req.input {
        EmbeddingInput::Single(s) => vec![s.clone()],
        EmbeddingInput::Multiple(vec) => vec.clone(),
    };

    // Get the appropriate engine for the requested model
    let mm = state.model_manager.read().await;
    let engine = match mm.get_model(Some(&model)) {
        Some(e) => e,
        None => mm.get_model(None).unwrap(),
    };
    drop(mm);

    let eng = engine.read().await;
    let mut data = Vec::new();
    let mut total_tokens = 0;

    for (idx, text) in inputs.iter().enumerate() {
        // Call the engine's embed method (mocked for now)
        let embedding = eng.embed(text).await;

        // Estimate tokens (rough: chars / 4)
        let tokens = (text.len() as i32 + 3) / 4;
        total_tokens += tokens;

        data.push(EmbeddingData {
            object: "embedding".into(),
            embedding,
            index: idx as u32,
        });
    }

    let response = EmbeddingsResponse {
        object: "list".into(),
        data,
        model: model.clone(),
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    tracing::info!(latency_ms = start.elapsed().as_millis(), "Embeddings done");
    (StatusCode::OK, Json(response))
}

/// Model load request
#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub name: String,
    pub path: String,
}

/// Model unload request
#[derive(Debug, Deserialize)]
pub struct ModelUnloadRequest {
    pub name: String,
}

/// Model reload request
#[derive(Debug, Deserialize)]
pub struct ModelReloadRequest {
    pub name: String,
    pub path: String,
}

/// Model info response
#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    pub name: String,
    pub path: String,
    pub state: String,
    pub loaded_at: Option<i64>,
    pub ready: bool,
}

/// Model load progress response
#[derive(Debug, Serialize)]
pub struct ModelLoadProgressResponse {
    pub model_name: String,
    pub progress: f32,
    pub state: String,
    pub message: String,
}

/// Model load response
#[derive(Debug, Serialize)]
pub struct ModelLoadResponse {
    pub status: String,
    pub model_name: String,
    pub message: String,
}

/// Load a new model (API trigger)
pub async fn model_load(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelLoadRequest>,
) -> (StatusCode, Json<ModelLoadResponse>) {
    tracing::info!(
        model_name = %req.name,
        model_path = %req.path,
        "Model load request"
    );

    let result = state.model_manager.write().await.load_model(&req.name, &req.path).await;

    match result {
        Ok(()) => {
            (StatusCode::OK, Json(ModelLoadResponse {
                status: "success".to_string(),
                model_name: req.name,
                message: "Model loaded successfully".to_string(),
            }))
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to load model");
            let status = if matches!(e, crate::error::AppError::ModelAlreadyLoaded(_)) {
                StatusCode::CONFLICT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(ModelLoadResponse {
                status: "error".to_string(),
                model_name: req.name,
                message: e.to_string(),
            }))
        }
    }
}

/// Unload a model (release memory)
pub async fn model_unload(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelUnloadRequest>,
) -> (StatusCode, Json<ModelLoadResponse>) {
    tracing::info!(model_name = %req.name, "Model unload request");

    let result = state.model_manager.write().await.unload_model(&req.name).await;

    match result {
        Ok(()) => {
            (StatusCode::OK, Json(ModelLoadResponse {
                status: "success".to_string(),
                model_name: req.name,
                message: "Model unloaded successfully".to_string(),
            }))
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to unload model");
            let status = if matches!(e, crate::error::AppError::CannotUnloadDefaultModel) {
                StatusCode::BAD_REQUEST
            } else if matches!(e, crate::error::AppError::ModelNotFound(_)) {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(ModelLoadResponse {
                status: "error".to_string(),
                model_name: req.name,
                message: e.to_string(),
            }))
        }
    }
}

/// Reload an existing model (hot reload)
pub async fn model_reload(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelReloadRequest>,
) -> (StatusCode, Json<ModelLoadResponse>) {
    tracing::info!(
        model_name = %req.name,
        new_path = %req.path,
        "Model reload request"
    );

    let result = state.model_manager.write().await.reload_model(&req.name, &req.path).await;

    match result {
        Ok(()) => {
            (StatusCode::OK, Json(ModelLoadResponse {
                status: "success".to_string(),
                model_name: req.name.clone(),
                message: format!("Model '{}' reloaded successfully", req.name),
            }))
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to reload model");
            let status = if matches!(e, crate::error::AppError::ModelNotFound(_)) {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(ModelLoadResponse {
                status: "error".to_string(),
                model_name: req.name,
                message: e.to_string(),
            }))
        }
    }
}

/// Get model loading progress
pub async fn model_load_progress(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelUnloadRequest>,
) -> (StatusCode, Json<ModelLoadProgressResponse>) {
    let progress = state.model_load_tracker.get_progress(&req.name).await;

    match progress {
        Some(p) => {
            let state_str = match p.state {
                crate::model::ModelState::Unloaded => "unloaded",
                crate::model::ModelState::Loading => "loading",
                crate::model::ModelState::Ready => "ready",
                crate::model::ModelState::Failed(_) => "failed",
            };
            (StatusCode::OK, Json(ModelLoadProgressResponse {
                model_name: p.model_name,
                progress: p.progress,
                state: state_str.to_string(),
                message: p.message,
            }))
        }
        None => {
            (StatusCode::NOT_FOUND, Json(ModelLoadProgressResponse {
                model_name: req.name,
                progress: 0.0,
                state: "not_found".to_string(),
                message: "Model loading operation not found".to_string(),
            }))
        }
    }
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
