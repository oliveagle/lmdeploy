use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, Sse},
    Json,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::server::AppState;

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: i32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
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

    let engine = state.engine.read().await;
    let text = engine.generate(&prompt, req.max_tokens.unwrap_or(512) as usize).await;

    let response = ChatCompletionsResponse {
        id: format!("chatcmpl-{}", uuid_simple()),
        object: "chat.completion".into(),
        created: unix_timestamp(),
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens: 0, // placeholder
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    tracing::info!(latency_ms = start.elapsed().as_millis(), "Chat completions done");
    (StatusCode::OK, Json(response))
}

pub async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let model = req.model.clone();
    let id = format!("chatcmpl-{}", uuid_simple());
    let created = unix_timestamp();

    tracing::info!(model = %model, "Chat completions stream request");

    let prompt = messages_to_prompt(&req.messages);
    let engine = state.engine.read().await;
    let chunks = engine.generate_stream(&prompt).await;

    let stream_id = id.clone();
    let stream_model = model.clone();
    let final_id = id.clone();
    let final_model = model.clone();

    let stream = chunks.map(move |chunk_text| {
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

    Sse::new(stream)
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
    let engine = state.engine.read().await;

    let text = engine.generate(&req.prompt, req.max_tokens.unwrap_or(512) as usize).await;

    let response = CompletionsResponse {
        id: format!("cmpl-{}", uuid_simple()),
        object: "text_completion".into(),
        created: unix_timestamp(),
        model,
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
    };

    (StatusCode::OK, Json(response))
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
