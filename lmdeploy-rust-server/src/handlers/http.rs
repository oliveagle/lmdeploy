use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
use tokio_stream::wrappers::ReceiverStream;

use crate::cache::compute_hash;
use crate::metrics::{StreamMetricsSnapshot, StreamRequestMetrics, EngineEventType};
use crate::model::GenerationParams;
use crate::server::{AppState, BatchItem, BatchStatsResponse};
use crate::turbomind_c::CompiledGrammar;

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
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
    /// Response format for structured output (guided decoding)
    pub response_format: Option<ResponseFormat>,
}

/// Response format for structured output (guided decoding).
/// Follows OpenAI's response_format specification.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Text response (no constraints)
    Text,
    /// JSON object output (constrained to valid JSON)
    JsonObject,
    /// JSON schema constrained output
    JsonSchema {
        json_schema: JsonSchemaSpec,
    },
    /// Regex constrained output
    RegexSchema {
        regex_schema: String,
    },
}

/// JSON schema specification for structured output.
/// Mirrors the OpenAI JSON schema format.
#[derive(Debug, Deserialize, Clone)]
pub struct JsonSchemaSpec {
    pub name: String,
    /// The actual JSON schema definition
    pub schema: serde_json::Value,
    pub strict: Option<bool>,
    pub description: Option<String>,
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
    /// Log probability information for each token (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Log probability information for a single token.
#[derive(Debug, Serialize, Clone)]
pub struct LogprobToken {
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
}

/// Top log probability entry.
#[derive(Debug, Serialize, Clone)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
}

/// Aggregated logprobs for a completion choice.
#[derive(Debug, Serialize, Clone)]
pub struct ChoiceLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f64>,
    pub top_logprobs: Vec<Option<TopLogprobEntry>>,
    pub top_tokens: Vec<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChunkUsage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
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
) -> Response {
    let model = req.model.clone();
    let start = std::time::Instant::now();

    // If stream: true, delegate to streaming endpoint
    if req.stream == Some(true) {
        let response = chat_completions_stream_impl(&state, &req, &model).await;
        return response;
    }

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
                        logprobs: None,
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

    tracing::info!(
        latency_ms = start.elapsed().as_millis(),
        "Chat completions done"
    );
    Json(response).into_response()
}

async fn chat_completions_stream_impl(
    state: &Arc<AppState>,
    req: &ChatCompletionsRequest,
    model: &str,
) -> Response {
    let id = format!("chatcmpl-{}", uuid_simple());
    let created = unix_timestamp();
    let keepalive_interval_ms = state
        .config
        .read()
        .await
        .server
        .stream_keepalive_interval_ms;

    tracing::info!(model = %model, "Chat completions stream request");

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

    let metrics = state.metrics.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(1024);
    let stream_id = id.clone();
    let stream_model = model.to_string();
    let final_id = id.clone();
    let final_model = model.to_string();
    let prompt_len = prompt.len();

    let grammar = response_format_to_grammar(&req.response_format);
    let params = GenerationParams::from_chat_request(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.max_tokens,
        req.seed,
        req.presence_penalty,
        req.frequency_penalty,
        req.stop.clone(),
        req.logprobs,
        req.top_logprobs,
        grammar,
    );

    // Spawn a task that generates tokens and sends them through the channel
    let metrics_inner = metrics.clone();
    let first_token_start = Instant::now();
    let prompt_for_task = prompt.clone();

    tokio::spawn(async move {
        // Initialize streaming metrics with QUEUED event
        let mut request_metrics = StreamRequestMetrics::new();
        request_metrics.record_event(EngineEventType::Queued);

        let eng = engine.read().await;

        // Record SCHEDULED event before starting inference
        request_metrics.record_event(EngineEventType::Scheduled);

        let chunks = eng.generate_stream(&prompt_for_task, params).await;
        futures::pin_mut!(chunks);

        let mut first_token_recorded = false;
        while let Some(chunk_text) = chunks.next().await {
            if !first_token_recorded {
                let latency_ms = first_token_start.elapsed().as_millis() as u64;
                metrics_inner.streams.record_stream_start(latency_ms);
                first_token_recorded = true;
            }

            metrics_inner.streams.record_chunk();

            // Update token timestamp in per-request metrics
            request_metrics.mark_token_generated();

            if tx.send(chunk_text).await.is_err() {
                tracing::info!("Client disconnected, stopping stream");
                break;
            }
        }

        // Log final request metrics
        tracing::debug!(
            token_timestamp_secs = request_metrics.token_timestamp_secs,
            num_events = request_metrics.engine_events.len(),
            "Request completed with metrics"
        );
    });

    // Convert channel into SSE stream with usage tracking
    // Optimized serialization: use pre-allocated buffers to reduce per-token allocations
    let stream = ReceiverStream::new(rx)
        .enumerate()
        .map(move |(idx, chunk_text)| {
            let completion_tokens = ((idx + 1) * 4).max(1) as i32;
            let prompt_tokens = (prompt_len / 4) as i32;

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
                usage: Some(ChunkUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                }),
            };
            // Optimized: use serde_json::to_writer to avoid intermediate allocation
            // Write directly into a pre-allocated buffer, then convert to String
            let mut buf = Vec::with_capacity(256);
            let mut writer = serde_json::Serializer::new(&mut buf);
            let json = if chunk.serialize(&mut writer).is_ok() {
                // SAFETY: serde_json always produces valid UTF-8
                unsafe { String::from_utf8_unchecked(buf) }
            } else {
                "{}".into()
            };
            let event = Event::default().event("chat.completion.chunk").data(json);
            Ok::<_, std::convert::Infallible>(event)
        })
        .chain(futures::stream::once(async move {
            let prompt_tokens = (prompt_len / 4) as i32;
            let final_chunk = ChatCompletionChunk {
                id: final_id,
                object: "chat.completion.chunk".into(),
                created,
                model: final_model,
                choices: vec![DeltaChoice {
                    index: 0,
                    delta: Delta {
                        content: None,
                        role: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: Some(ChunkUsage {
                    prompt_tokens,
                    completion_tokens: 0,
                    total_tokens: prompt_tokens,
                }),
            };
            // Optimized: serialize directly into buffer for efficiency
            let mut buf = Vec::with_capacity(256);
            let mut writer = serde_json::Serializer::new(&mut buf);
            let json = if final_chunk.serialize(&mut writer).is_ok() {
                // SAFETY: serde_json always produces valid UTF-8
                unsafe { String::from_utf8_unchecked(buf) }
            } else {
                "{}".into()
            };
            Ok(Event::default().event("chat.completion.chunk").data(json))
        }));

    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_millis(keepalive_interval_ms)),
        )
        .into_response()
}

async fn fallback_chat_completion(
    state: &Arc<AppState>,
    req: &ChatCompletionsRequest,
    model: &str,
) -> ChatCompletionsResponse {
    let prompt = messages_to_prompt(&req.messages);
    let need_logprobs = req.logprobs.unwrap_or(false) || req.top_logprobs.unwrap_or(0) > 0;

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

    let grammar = response_format_to_grammar(&req.response_format);
    let params = GenerationParams::from_chat_request(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.max_tokens,
        req.seed,
        req.presence_penalty,
        req.frequency_penalty,
        req.stop.clone(),
        req.logprobs,
        req.top_logprobs,
        grammar,
    );

    let eng = engine.read().await;
    let (text, logprobs) = if need_logprobs {
        let (t, _nt, _el, lp) = eng.generate_with_logprobs(&prompt, params).await;
        (t, lp)
    } else {
        let t = eng.generate(&prompt, params).await;
        (t, None)
    };

    ChatCompletionsResponse {
        id: format!("chatcmpl-{}", uuid_simple()),
        object: "chat.completion".into(),
        created: unix_timestamp(),
        model: model.to_string(),
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
    }
}

pub async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response {
    let model = req.model.clone();
    chat_completions_stream_impl(&state, &req, &model).await
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
    pub response_format: Option<ResponseFormat>,
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
                top_k: None,
                min_p: None,
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
                response_format: req.response_format.clone(),
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
                        choices: chat_resp
                            .choices
                            .iter()
                            .map(|c| CompletionChoice {
                                text: c.message.content.clone(),
                                index: c.index,
                                finish_reason: c.finish_reason.clone(),
                            })
                            .collect(),
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

    let params = GenerationParams {
        max_tokens: req.max_tokens.map(|t| t as usize),
        temperature: req.temperature,
        top_p: None,
        top_k: None,
        ..Default::default()
    };

    let eng = engine.read().await;
    let text = eng.generate(prompt_text, params).await;

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

pub async fn list_models(State(state): State<Arc<AppState>>) -> (StatusCode, Json<ModelsResponse>) {
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

    (
        StatusCode::OK,
        Json(ModelsResponse {
            object: "list".into(),
            data,
        }),
    )
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
    // Optimized: pre-allocate with capacity hint to reduce reallocations
    let mut choices = Vec::with_capacity(req.messages.len());
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;

    let params = GenerationParams {
        max_tokens: req.max_tokens.map(|t| t as usize),
        temperature: req.temperature,
        top_p: req.top_p,
        ..Default::default()
    };

    for (idx, messages) in req.messages.iter().enumerate() {
        let prompt = messages_to_prompt(messages);
        let text = eng.generate(&prompt, params.clone()).await;

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

    tracing::info!(
        latency_ms = start.elapsed().as_millis(),
        "Batch chat completions done"
    );
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
    // Optimized: pre-allocate with capacity hint to reduce reallocations
    let mut choices = Vec::with_capacity(req.prompts.len());
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;

    let params = GenerationParams {
        max_tokens: req.max_tokens.map(|t| t as usize),
        temperature: req.temperature,
        ..Default::default()
    };

    for (idx, prompt) in req.prompts.iter().enumerate() {
        let text = eng.generate(prompt, params.clone()).await;

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

    tracing::info!(
        latency_ms = start.elapsed().as_millis(),
        "Batch completions done"
    );
    (StatusCode::OK, Json(response))
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

pub async fn health_check() -> (StatusCode, Json<HealthResponse>) {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }),
    )
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

    let state_clone = state.clone();
    let token_ids = state
        .tokenizer_cache
        .get_or_tokenize(&req.text, |text| {
            let owned = text.to_string();
            async move {
                let mgr = state_clone.model_manager.read().await;
                if let Some(tokenizer) = mgr.get_default_tokenizer().await {
                    tokenizer.encode(&owned, false, false).map_err(|e| {
                        crate::error::AppError::Other(format!("Tokenization failed: {}", e))
                    })
                } else {
                    Err(crate::error::AppError::Other(
                        "No tokenizer available - model may not be loaded yet".to_string(),
                    ))
                }
            }
        })
        .await
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "Tokenization failed, falling back to mock");
            req.text.chars().map(|c| c as u32).collect()
        });

    let cached = start.elapsed().as_millis() < 1;

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

/// Request to set prefix caching mode
#[derive(Debug, Deserialize)]
pub struct SetPrefixCacheRequest {
    pub enabled: bool,
}

#[derive(Debug, Serialize)]
pub struct SetPrefixCacheResponse {
    pub status: String,
    pub message: String,
    pub prefix_cache_enabled: bool,
}

/// Set prefix caching mode (takes effect on next model reload)
pub async fn set_prefix_cache(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetPrefixCacheRequest>,
) -> (StatusCode, Json<SetPrefixCacheResponse>) {
    let mut config = state.config.write().await;
    config.model.prefix_cache_enabled = req.enabled;

    tracing::info!(
        prefix_cache_enabled = req.enabled,
        "Prefix caching mode updated via API"
    );

    (
        StatusCode::OK,
        Json(SetPrefixCacheResponse {
            status: "ok".into(),
            message: if req.enabled {
                "Prefix caching enabled. Reload model to apply changes.".into()
            } else {
                "Prefix caching disabled. Reload model to apply changes.".into()
            },
            prefix_cache_enabled: req.enabled,
        }),
    )
}

#[derive(Debug, Serialize)]
pub struct PrefixCacheStatusResponse {
    pub prefix_cache_enabled: bool,
    pub message: String,
}

/// Get current prefix caching status
pub async fn get_prefix_cache_status(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<PrefixCacheStatusResponse>) {
    let config = state.config.read().await;
    let enabled = config.model.prefix_cache_enabled;

    (
        StatusCode::OK,
        Json(PrefixCacheStatusResponse {
            prefix_cache_enabled: enabled,
            message: if enabled {
                "Prefix caching is enabled".into()
            } else {
                "Prefix caching is disabled".into()
            },
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
// Rate Limiting Status
// ============================================================================

/// Rate limit status response
#[derive(Debug, Serialize)]
pub struct RateLimitStatusResponse {
    pub rate_limiting_enabled: bool,
    pub global_requests_per_second: u32,
    pub global_burst_size: u32,
    pub per_ip_enabled: bool,
    pub per_ip_requests_per_second: u32,
    pub per_ip_burst_size: u32,
    pub global_total_requests: u64,
    pub global_rate_limited: u64,
    pub per_ip_total_requests: u64,
    pub per_ip_rate_limited: u64,
    pub tracked_ips: usize,
}

/// Rate limit status endpoint
pub async fn rate_limit_status(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<RateLimitStatusResponse>) {
    let config = state.config.read().await;
    let rl_config = &config.server.rate_limit;

    let (global_total, global_limited, per_ip_total, per_ip_limited, tracked_ips) =
        if let Some(rl) = &state.global_rate_limiter {
            let gt = rl.total_requests();
            let gl = rl.rate_limited();
            if let Some(per_ip) = &state.per_ip_rate_limiter {
                (
                    gt,
                    gl,
                    per_ip.total_requests(),
                    per_ip.rate_limited(),
                    per_ip.tracked_ips(),
                )
            } else {
                (gt, gl, 0, 0, 0)
            }
        } else {
            (0, 0, 0, 0, 0)
        };

    (
        StatusCode::OK,
        Json(RateLimitStatusResponse {
            rate_limiting_enabled: rl_config.enabled,
            global_requests_per_second: rl_config.requests_per_second,
            global_burst_size: rl_config.burst_size,
            per_ip_enabled: rl_config.per_ip.enabled,
            per_ip_requests_per_second: rl_config.per_ip.requests_per_second,
            per_ip_burst_size: rl_config.per_ip.burst_size,
            global_total_requests: global_total,
            global_rate_limited: global_limited,
            per_ip_total_requests: per_ip_total,
            per_ip_rate_limited: per_ip_limited,
            tracked_ips,
        }),
    )
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
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| "default-embedding-model".to_string());

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
    // Optimized: pre-allocate with capacity hint to reduce reallocations
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0;
    let dimensions = req.dimensions.map(|d| d as usize);

    for (idx, text) in inputs.iter().enumerate() {
        // Call the engine's embed method (mocked for now)
        let embedding = eng.embed(text, dimensions).await;

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

    let result = state
        .model_manager
        .write()
        .await
        .load_model(&req.name, &req.path)
        .await;

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(ModelLoadResponse {
                status: "success".to_string(),
                model_name: req.name,
                message: "Model loaded successfully".to_string(),
            }),
        ),
        Err(e) => {
            tracing::error!(error = %e, "Failed to load model");
            let status = if matches!(e, crate::error::AppError::ModelAlreadyLoaded(_)) {
                StatusCode::CONFLICT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ModelLoadResponse {
                    status: "error".to_string(),
                    model_name: req.name,
                    message: e.to_string(),
                }),
            )
        }
    }
}

/// Unload a model (release memory)
pub async fn model_unload(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelUnloadRequest>,
) -> (StatusCode, Json<ModelLoadResponse>) {
    tracing::info!(model_name = %req.name, "Model unload request");

    let result = state
        .model_manager
        .write()
        .await
        .unload_model(&req.name)
        .await;

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(ModelLoadResponse {
                status: "success".to_string(),
                model_name: req.name,
                message: "Model unloaded successfully".to_string(),
            }),
        ),
        Err(e) => {
            tracing::error!(error = %e, "Failed to unload model");
            let status = if matches!(e, crate::error::AppError::CannotUnloadDefaultModel) {
                StatusCode::BAD_REQUEST
            } else if matches!(e, crate::error::AppError::ModelNotFound(_)) {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ModelLoadResponse {
                    status: "error".to_string(),
                    model_name: req.name,
                    message: e.to_string(),
                }),
            )
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

    let result = state
        .model_manager
        .write()
        .await
        .reload_model(&req.name, &req.path)
        .await;

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(ModelLoadResponse {
                status: "success".to_string(),
                model_name: req.name.clone(),
                message: format!("Model '{}' reloaded successfully", req.name),
            }),
        ),
        Err(e) => {
            tracing::error!(error = %e, "Failed to reload model");
            let status = if matches!(e, crate::error::AppError::ModelNotFound(_)) {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ModelLoadResponse {
                    status: "error".to_string(),
                    model_name: req.name,
                    message: e.to_string(),
                }),
            )
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
            (
                StatusCode::OK,
                Json(ModelLoadProgressResponse {
                    model_name: p.model_name,
                    progress: p.progress,
                    state: state_str.to_string(),
                    message: p.message,
                }),
            )
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(ModelLoadProgressResponse {
                model_name: req.name,
                progress: 0.0,
                state: "not_found".to_string(),
                message: "Model loading operation not found".to_string(),
            }),
        ),
    }
}

/// Convert ResponseFormat to a CompiledGrammar for guided decoding.
///
/// This function converts the OpenAI-compatible `response_format` parameter
/// into a grammar constraint that can be applied to the C++ engine.
///
/// # Arguments
/// * `response_format` - The optional response format from the request
///
/// # Returns
/// * `Some(Arc<CompiledGrammar>)` - If a valid grammar constraint is specified
/// * `None` - If no constraint should be applied (text mode or error)
pub fn response_format_to_grammar(response_format: &Option<ResponseFormat>) -> Option<Arc<CompiledGrammar>> {
    match response_format {
        Some(ResponseFormat::Text) => None,
        Some(ResponseFormat::JsonObject) => {
            // No custom schema - use built-in JSON grammar
            Some(Arc::new(CompiledGrammar::builtin_json()))
        }
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            // Custom JSON schema - serialize and compile
            let schema_str = json_schema.schema.to_string();
            match CompiledGrammar::from_json_schema(&schema_str) {
                Ok(grammar) => {
                    tracing::debug!(schema = %json_schema.name, "Created JSON schema grammar");
                    Some(Arc::new(grammar))
                }
                Err(e) => {
                    tracing::warn!(error = ?e, schema = %json_schema.name, "Failed to create JSON schema grammar, falling back to builtin JSON");
                    Some(Arc::new(CompiledGrammar::builtin_json()))
                }
            }
        }
        Some(ResponseFormat::RegexSchema { regex_schema }) => {
            // Regex constraint - compile via xgrammar
            match CompiledGrammar::from_regex(regex_schema) {
                Ok(grammar) => {
                    tracing::debug!("Created regex grammar");
                    Some(Arc::new(grammar))
                }
                Err(e) => {
                    tracing::warn!(error = ?e, "Failed to create regex grammar");
                    None
                }
            }
        }
        None => None,
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
