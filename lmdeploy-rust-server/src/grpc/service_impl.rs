use std::sync::Arc;
use std::time::Instant;

use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status, Streaming};

use crate::cache::TokenizeCache;
use crate::model::ModelManager;
use crate::error::AppError;

use super::lmdeploy::v1::{
    lm_deploy_service_server::LmDeployService,
    generate_stream_response,
    GenerateRequest, GenerateResponse,
    GenerateStreamResponse, BatchGenerateRequest, BatchGenerateResponse,
    StreamChunk, HealthRequest, HealthResponse,
    ModelInfoRequest, ModelInfoResponse,
    TokenizeRequest, TokenizeResponse,
};

#[derive(Clone)]
pub struct LmDeployServiceImpl {
    pub version: String,
    pub tokenizer_cache: Arc<TokenizeCache>,
    pub model_manager: Arc<RwLock<ModelManager>>,
}

#[tonic::async_trait]
impl LmDeployService for LmDeployServiceImpl {
    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let req = request.into_inner();

        tracing::info!(prompt_len = req.prompt.len(), "gRPC Generate request");

        let resp = GenerateResponse {
            text: format!("Mock response for: {}", &req.prompt[..req.prompt.len().min(50)]),
            token_ids: vec![],
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: 0.0,
            error: String::new(),
            latency_ms: 0.0,
            tokens_per_second: 0.0,
        };

        Ok(Response::new(resp))
    }

    type GenerateStreamStream = ReceiverStream<Result<GenerateStreamResponse, Status>>;

    async fn generate_stream(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let req = request.into_inner();

        tracing::info!(prompt_len = req.prompt.len(), "gRPC GenerateStream request");

        let words: Vec<String> = req.prompt.split_whitespace().map(String::from).collect();
        let (tx, rx) = tokio::sync::mpsc::channel(1024);
        let timeout_secs = 600u64; // Default stream timeout (matches config default)

        tokio::spawn(async move {
            let mut first_token_recorded = false;
            let first_token_start = Instant::now();
            let timeout_duration = std::time::Duration::from_secs(timeout_secs);

            for word in words {
                if first_token_start.elapsed() > timeout_duration {
                    tracing::warn!("gRPC stream timeout after {}s", timeout_secs);
                    break;
                }

                let latency_ms = first_token_start.elapsed().as_millis() as u64;
                if !first_token_recorded {
                    tracing::info!(first_token_latency_ms = latency_ms, "First token latency recorded (gRPC stream)");
                    first_token_recorded = true;
                }

                let chunk = GenerateStreamResponse {
                    payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                        text: word.clone(),
                        token_id: 0,
                        is_final: false,
                    })),
                };

                if tx.send(Ok(chunk)).await.is_err() {
                    tracing::info!("Client disconnected, stopping stream");
                    break;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }

            let final_chunk = GenerateStreamResponse {
                payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                    text: "[END]".to_string(),
                    token_id: 0,
                    is_final: true,
                })),
            };

            let _ = tx.send(Ok(final_chunk)).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    type GenerateBidirectionalStream = ReceiverStream<Result<GenerateStreamResponse, Status>>;

        async fn generate_bidirectional(
        &self,
        request: Request<Streaming<GenerateRequest>>,
    ) -> Result<Response<Self::GenerateBidirectionalStream>, Status> {
        let mut stream = request.into_inner();

        tracing::info!("gRPC GenerateBidirectional request");

        let (tx, rx) = tokio::sync::mpsc::channel(1024);
        let timeout_secs = 600u64; // Default stream timeout (matches config default)
        let timeout_duration = std::time::Duration::from_secs(timeout_secs);

        tokio::spawn(async move {
            let mut last_activity = Instant::now();
            let mut stream_ended = false;

            while !stream_ended && last_activity.elapsed() < timeout_duration {
                tokio::select! {
                    biased;

                    _ = tokio::time::sleep(timeout_duration.saturating_sub(last_activity.elapsed())) => {
                        if last_activity.elapsed() >= timeout_duration {
                            tracing::warn!("gRPC bidirectional stream timeout after {}s of inactivity", timeout_secs);
                            break;
                        }
                    }
                    request_result = stream.next() => {
                        match request_result {
                            Some(Ok(req)) => {
                                last_activity = Instant::now();

                                tracing::info!(
                                    prompt_len = req.prompt.len(),
                                    "Bidirectional stream request"
                                );

                                let words: Vec<String> =
                                    req.prompt.split_whitespace().map(String::from).collect();

                                for word in words {
                                    let chunk = GenerateStreamResponse {
                                        payload: Some(
                                            generate_stream_response::Payload::Chunk(StreamChunk {
                                                text: word.clone(),
                                                token_id: 0,
                                                is_final: false,
                                            }),
                                        ),
                                    };

                                    if tx.send(Ok(chunk)).await.is_err() {
                                        tracing::info!("Client disconnected, stopping stream");
                                        return;
                                    }

                                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                                }

                                let final_chunk = GenerateStreamResponse {
                                    payload: Some(
                                        generate_stream_response::Payload::Chunk(StreamChunk {
                                            text: "[END]".to_string(),
                                            token_id: 0,
                                            is_final: true,
                                        }),
                                    ),
                                };

                                if tx.send(Ok(final_chunk)).await.is_err() {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                tracing::error!("Error in bidirectional stream: {:?}", e);
                                let _ = tx
                                    .send(Err(Status::invalid_argument(format!("Invalid request: {}", e))))
                                    .await;
                                break;
                            }
                            None => {
                                tracing::info!("Client stream ended");
                                stream_ended = true;
                            }
                        }
                    }
                }
            }

            tracing::info!("Bidirectional stream completed");
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn generate_batch(
        &self,
        request: Request<BatchGenerateRequest>,
    ) -> Result<Response<BatchGenerateResponse>, Status> {
        let req = request.into_inner();

        tracing::info!(num_requests = req.requests.len(), "gRPC GenerateBatch request");

        let mut responses = Vec::with_capacity(req.requests.len());

        for gen_req in &req.requests {
            let resp = GenerateResponse {
                text: format!(
                    "Mock batch response for: {}",
                    &gen_req.prompt[..gen_req.prompt.len().min(50)]
                ),
                token_ids: vec![],
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: 0.0,
                error: String::new(),
                latency_ms: 0.0,
                tokens_per_second: 0.0,
            };

            responses.push(resp);
        }

        let resp = BatchGenerateResponse {
            responses,
            total_latency_ms: 0.0,
            throughput_tokens_per_second: 0.0,
        };

        Ok(Response::new(resp))
    }

    async fn tokenize(
        &self,
        request: Request<TokenizeRequest>,
    ) -> Result<Response<TokenizeResponse>, Status> {
        let req = request.into_inner();

        tracing::info!(text_len = req.text.len(), "gRPC Tokenize request");

        let hash = crate::cache::compute_hash(&req.text);
        let start = std::time::Instant::now();

        let token_ids = self
            .tokenizer_cache
            .get_or_tokenize(&req.text, |text| {
                let owned = text.to_string();
                let mm = self.model_manager.clone();
                async move {
                    let manager = mm.read().await;
                    if let Some(tokenizer) = manager.get_default_tokenizer().await {
                        tokenizer.encode(&owned, false, false)
                            .map_err(|e| AppError::Other(format!("Tokenization failed: {}", e)))
                    } else {
                        Err(AppError::Other("No tokenizer available".to_string()))
                    }
                }
            })
            .await
            .unwrap_or_else(|e| {
                tracing::error!(error = %e, "gRPC tokenization failed, falling back to mock");
                req.text.chars().map(|c| c as u32).collect()
            });

        let length = token_ids.len();
        let latency_ms = start.elapsed().as_millis() as f64;

        // Decode tokens back to string representation
        let tokens = {
            let manager = self.model_manager.read().await;
            if let Some(tokenizer) = manager.get_default_tokenizer().await {
                token_ids.iter()
                    .map(|&id| tokenizer.id_to_token(id).unwrap_or_else(|| format!("<id:{}>", id)))
                    .collect()
            } else {
                req.text.split_whitespace().map(String::from).collect()
            }
        };

        let resp = TokenizeResponse {
            token_ids,
            tokens,
            length: length as i32,
            hash,
        };

        tracing::info!(
            token_count = length,
            latency_ms = latency_ms,
            "gRPC Tokenize response"
        );

        Ok(Response::new(resp))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let resp = HealthResponse {
            status: "ok".into(),
            version: self.version.clone(),
            model: "mock-model".into(),
            uptime_seconds: 0,
        };

        Ok(Response::new(resp))
    }

    async fn model_info(
        &self,
        _request: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        let resp = ModelInfoResponse {
            model_name: "lmdeploy-mock".into(),
            model_path: "/mock/path".into(),
            max_context_length: 8192,
            vocab_size: 151936,
            supports_streaming: true,
            capabilities: vec![
                "generate".into(),
                "stream".into(),
                "batch".into(),
                "tokenize".into(),
            ],
        };

        Ok(Response::new(resp))
    }
}