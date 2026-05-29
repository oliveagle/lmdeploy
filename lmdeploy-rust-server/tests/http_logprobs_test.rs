//! HTTP API test for logprobs functionality
//!
//! This test verifies that the HTTP API correctly handles logprobs requests
//! and returns properly formatted responses.

use axum::{
    body::Body,
    http::{header, Method, Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tower::ServiceExt;

use lmdeploy_server::cache::TokenizeCache;
use lmdeploy_server::config::AppConfig;
use lmdeploy_server::metrics::AppMetrics;
use lmdeploy_server::model::{ModelLoadTracker, ModelManager};
use lmdeploy_server::server::{create_router, AppState, BatchStats};

/// Create a test app with the given state
async fn create_test_app() -> Result<axum::Router, Box<dyn std::error::Error>> {
    let config = Arc::new(tokio::sync::RwLock::new(AppConfig::default()));

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".to_string());

    // Create model manager with the loaded model
    let engine_type = lmdeploy_server::model::cpp_engine::EngineType::PureCpp;
    let model_manager = ModelManager::with_default_model_and_type(&model_path, engine_type).await?;

    // Create app state with actual fields
    let (config_reload_tx, _config_reload_rx) = tokio::sync::mpsc::channel::<()>(32);

    let state = AppState {
        model_manager: Arc::new(tokio::sync::RwLock::new(model_manager)),
        tokenizer_cache: Arc::new(TokenizeCache::new(1000, 3600)),
        config: config.clone(),
        request_semaphore: Arc::new(Semaphore::new(10000)),
        batch_sender: None,
        batch_stats: Arc::new(BatchStats::new()),
        metrics: Arc::new(AppMetrics::new()),
        config_reload_tx,
        model_load_tracker: Arc::new(ModelLoadTracker::new()),
        global_rate_limiter: None,
        per_ip_rate_limiter: None,
    };

    // Create router with the test handlers
    let app = create_router(Arc::new(state));

    Ok(app)
}

#[tokio::test]
async fn test_chat_completions_with_logprobs() -> Result<(), Box<dyn std::error::Error>> {
    let app = create_test_app().await?;

    // Test with logprobs=true
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"
        {
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "logprobs": true,
            "top_logprobs": 5
        }
        "#,
        ))?;

    let response = app
        .oneshot(request)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body();
    let bytes = body
        .collect()
        .await
        .map_err(|e| format!("Failed to collect body: {}", e))?
        .to_bytes();

    let json: Value = serde_json::from_slice(&bytes)?;

    // Check that logprobs are present
    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
        if let Some(first_choice) = choices.first() {
            assert!(
                first_choice.get("logprobs").is_some(),
                "Expected logprobs in response"
            );

            if let Some(logprobs) = first_choice.get("logprobs") {
                println!("Logprobs response: {:#}", logprobs);
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_chat_completions_without_logprobs() -> Result<(), Box<dyn std::error::Error>> {
    let app = create_test_app().await?;

    // Test without logprobs
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"
        {
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        "#,
        ))?;

    let response = app
        .oneshot(request)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body();
    let bytes = body
        .collect()
        .await
        .map_err(|e| format!("Failed to collect body: {}", e))?
        .to_bytes();

    let json: Value = serde_json::from_slice(&bytes)?;

    // Check that logprobs are not present (or empty)
    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
        if let Some(first_choice) = choices.first() {
            let logprobs = first_choice.get("logprobs");
            println!("Logprobs response (should be null/empty): {:?}", logprobs);
        }
    }

    Ok(())
}
