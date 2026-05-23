use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::cache::TokenizeCache;
use crate::error::AppError;
use crate::model::ModelManager;
use crate::model::{BatchItem, BatchResult, GenerationParams, ModelEngine};

use super::lmdeploy::v1::{
    generate_stream_response, lm_deploy_service_server::LmDeployService, BatchGenerateRequest,
    BatchGenerateResponse, GenerateRequest, GenerateResponse, GenerateStreamResponse,
    HealthRequest, HealthResponse, ModelInfoRequest, ModelInfoResponse, StreamChunk,
    TokenizeRequest, TokenizeResponse,
};

/// Helper to get the default engine from model manager
fn get_default_engine(
    manager: &Arc<RwLock<ModelManager>>,
) -> Option<std::sync::Arc<tokio::sync::RwLock<ModelEngine>>> {
    // Note: This returns a clone of the Arc, not a borrow to the internal model
    manager.blocking_read().get_model(None)
}

#[derive(Clone)]
pub struct LmDeployServiceImpl {
    pub version: String,
    pub tokenizer_cache: Arc<TokenizeCache>,
    pub model_manager: Arc<RwLock<ModelManager>>,
    /// Server start time for uptime calculation
    start_time: std::time::Instant,
}

impl LmDeployServiceImpl {
    pub fn new(
        version: String,
        tokenizer_cache: Arc<TokenizeCache>,
        model_manager: Arc<RwLock<ModelManager>>,
    ) -> Self {
        Self {
            version,
            tokenizer_cache,
            model_manager,
            start_time: std::time::Instant::now(),
        }
    }
}

#[tonic::async_trait]
impl LmDeployService for LmDeployServiceImpl {
    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let req = request.into_inner();

        tracing::info!(prompt_len = req.prompt.len(), "gRPC Generate request");

        // Get the engine from model manager
        let manager = self.model_manager.read().await;
        let engine_ref = match get_default_engine(&manager) {
            Some(e) => e,
            None => {
                let resp = GenerateResponse {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    finish_reason: 2.0,
                    error: "No model loaded".to_string(),
                    latency_ms: 0.0,
                    tokens_per_second: 0.0,
                };
                return Ok(Response::new(resp));
            }
        };

        let engine = engine_ref.read().await;
        let eng = &*engine;

        let params = GenerationParams::from_grpc_request(
            if req.max_tokens > 0 { Some(req.max_tokens as usize) } else { None },
            if req.temperature > 0.0 { Some(req.temperature) } else { None },
            if req.top_p > 0.0 { Some(req.top_p) } else { None },
            if req.top_k > 0 { Some(req.top_k) } else { None },
            if req.repetition_penalty > 0 { Some(req.repetition_penalty as f32) } else { None },
            if req.seed > 0 { Some(req.seed as u64) } else { None },
        );

        let (text, num_tokens, elapsed_ms) = eng
            .generate_with_metrics(&req.prompt, params)
            .await;

        // Calculate tokens per second
        let tokens_per_second = if elapsed_ms > 0.0 && num_tokens > 0 {
            (num_tokens as f64 / elapsed_ms) * 1000.0
        } else {
            0.0
        };

        // Count prompt tokens if we have a tokenizer
        let prompt_tokens = manager
            .get_default_tokenizer()
            .await
            .and_then(|t| t.encode(&req.prompt, false, false).ok())
            .map(|ids| ids.len())
            .unwrap_or(0);

        let is_empty = text.is_empty();
        let resp = GenerateResponse {
            text,
            token_ids: vec![],
            prompt_tokens: prompt_tokens as i32,
            completion_tokens: num_tokens as i32,
            finish_reason: if !is_empty { 0.0 } else { 2.0 },
            error: String::new(),
            latency_ms: elapsed_ms as f32,
            tokens_per_second: tokens_per_second as f32,
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

        let (tx, rx) = tokio::sync::mpsc::channel(1024);

        // Clone necessary data for the spawned task
        let manager = Arc::clone(&self.model_manager);
        let params = GenerationParams::from_grpc_request(
            if req.max_tokens > 0 { Some(req.max_tokens as usize) } else { None },
            if req.temperature > 0.0 { Some(req.temperature) } else { None },
            if req.top_p > 0.0 { Some(req.top_p) } else { None },
            if req.top_k > 0 { Some(req.top_k) } else { None },
            if req.repetition_penalty > 0 { Some(req.repetition_penalty as f32) } else { None },
            if req.seed > 0 { Some(req.seed as u64) } else { None },
        );
        let prompt = req.prompt;

        tokio::spawn(async move {
            let first_token_start = Instant::now();
            let mut first_token_recorded = false;

            // Get the engine
            let Some(engine_ref) = get_default_engine(&manager) else {
                let _ = tx.send(Ok(GenerateStreamResponse {
                    payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                        text: "[ERROR: No model loaded]".to_string(),
                        token_id: 0,
                        is_final: true,
                    })),
                })).await;
                return;
            };

            // Lock the engine for the duration of streaming
            let engine_guard = engine_ref.read().await;
            let engine = &*engine_guard;

            // Use the engine's streaming method
            let mut stream = engine
                .generate_stream(&prompt, params)
                .await;

            // Process tokens from the stream
            while let Ok(Some(token_result)) = tokio::time::timeout(
                std::time::Duration::from_secs(600),
                stream.next(),
            ).await
            {
                // stream.next() returns String directly (not Result<String, Error>)
                let token = token_result;
                let latency_ms = first_token_start.elapsed().as_millis() as u64;
                if !first_token_recorded {
                    tracing::info!(
                        first_token_latency_ms = latency_ms,
                        "First token latency recorded (gRPC stream)"
                    );
                    first_token_recorded = true;
                }

                let chunk = GenerateStreamResponse {
                    payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                        text: token,
                        token_id: 0,
                        is_final: false,
                    })),
                };

                if tx.send(Ok(chunk)).await.is_err() {
                    tracing::info!("Client disconnected, stopping stream");
                    return;
                }
            }

            // Send final chunk
            let final_chunk = GenerateStreamResponse {
                payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                    text: "[DONE]".to_string(),
                    token_id: 0,
                    is_final: true,
                })),
            };

            let _ = tx.send(Ok(final_chunk)).await;
            tracing::info!("gRPC stream completed");
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

        // Clone necessary data for the spawned task
        let manager = Arc::clone(&self.model_manager);
        let timeout_duration = std::time::Duration::from_secs(600);

        tokio::spawn(async move {
            let mut last_activity = Instant::now();
            let mut stream_ended = false;

            while !stream_ended && last_activity.elapsed() < timeout_duration {
                tokio::select! {
                    biased;

                    _ = tokio::time::sleep(timeout_duration.saturating_sub(last_activity.elapsed())) => {
                        if last_activity.elapsed() >= timeout_duration {
                            tracing::warn!("gRPC bidirectional stream timeout after inactivity");
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

                                // Get the engine for this request
                                let Some(engine_ref) = get_default_engine(&manager) else {
                                    let _ = tx.send(Ok(GenerateStreamResponse {
                                        payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                                            text: "[ERROR: No model loaded]".to_string(),
                                            token_id: 0,
                                            is_final: true,
                                        })),
                                    })).await;
                                    continue;
                                };
                                let engine_guard = engine_ref.read().await;
                                let engine = &*engine_guard;

                                // Build generation params
                                let params = GenerationParams::from_grpc_request(
                                    if req.max_tokens > 0 { Some(req.max_tokens as usize) } else { None },
                                    if req.temperature > 0.0 { Some(req.temperature) } else { None },
                                    if req.top_p > 0.0 { Some(req.top_p) } else { None },
                                    if req.top_k > 0 { Some(req.top_k) } else { None },
                                    if req.repetition_penalty > 0 { Some(req.repetition_penalty as f32) } else { None },
                                    if req.seed > 0 { Some(req.seed as u64) } else { None },
                                );

                                // Process streaming response
                                let mut token_stream = engine
                                    .generate_stream(&req.prompt, params)
                                    .await;

                                while let Ok(Some(token)) = tokio::time::timeout(
                                    std::time::Duration::from_secs(300),
                                    token_stream.next(),
                                ).await
                                {
                                    let chunk = GenerateStreamResponse {
                                        payload: Some(
                                            generate_stream_response::Payload::Chunk(StreamChunk {
                                                text: token,
                                                token_id: 0,
                                                is_final: false,
                                            })),
                                    };

                                    if tx.send(Ok(chunk)).await.is_err() {
                                        tracing::info!("Client disconnected, stopping stream");
                                        return;
                                    }
                                }

                                // Send final marker
                                let final_chunk = GenerateStreamResponse {
                                    payload: Some(
                                        generate_stream_response::Payload::Chunk(StreamChunk {
                                            text: "[DONE]".to_string(),
                                            token_id: 0,
                                            is_final: true,
                                        })),
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

        tracing::info!(
            num_requests = req.requests.len(),
            "gRPC GenerateBatch request"
        );

        let manager = self.model_manager.read().await;
        let engine_ref = match get_default_engine(&manager) {
            Some(e) => e,
            None => {
                let error_resp = BatchGenerateResponse {
                    responses: vec![],
                    total_latency_ms: 0.0,
                    throughput_tokens_per_second: 0.0,
                };
                return Ok(Response::new(error_resp));
            }
        };

        let engine = engine_ref.read().await;

        // Convert gRPC requests to BatchItem
        let batch_items: Vec<BatchItem> = req
            .requests
            .into_iter()
            .enumerate()
            .map(|(idx, gen_req)| BatchItem {
                request_id: idx as u64,
                prompt: gen_req.prompt,
                params: GenerationParams::from_grpc_request(
                    if gen_req.max_tokens > 0 { Some(gen_req.max_tokens as usize) } else { None },
                    if gen_req.temperature > 0.0 { Some(gen_req.temperature) } else { None },
                    if gen_req.top_p > 0.0 { Some(gen_req.top_p) } else { None },
                    if gen_req.top_k > 0 { Some(gen_req.top_k) } else { None },
                    if gen_req.repetition_penalty > 0 { Some(gen_req.repetition_penalty as f32) } else { None },
                    if gen_req.seed > 0 { Some(gen_req.seed as u64) } else { None },
                ),
                need_logprobs: false,
            })
            .collect();

        // Process batch requests sequentially (the engine handles internal parallelism)
        let mut batch_results = Vec::with_capacity(batch_items.len());
        for item in batch_items {
            let (text, num_tokens, elapsed_ms) = engine.generate_with_metrics(&item.prompt, item.params).await;
            batch_results.push(BatchResult {
                request_id: item.request_id,
                text,
                num_tokens,
                elapsed_ms,
                logprobs: None,
                error: None,
            });
        }

        let total_latency_ms = batch_results
            .iter()
            .map(|r| r.elapsed_ms)
            .sum::<f64>();

        let total_tokens: usize = batch_results
            .iter()
            .map(|r| r.num_tokens)
            .sum();

        let throughput_tokens_per_second = if total_latency_ms > 0.0 {
            (total_tokens as f64 / total_latency_ms) * 1000.0
        } else {
            0.0
        };

        // Convert BatchResults to GenerateResponse
        let responses: Vec<GenerateResponse> = batch_results
            .into_iter()
            .map(|r| GenerateResponse {
                text: r.text,
                token_ids: vec![],
                prompt_tokens: 0,
                completion_tokens: r.num_tokens as i32,
                finish_reason: if r.error.is_none() { 0.0 } else { 2.0 },
                error: r.error.unwrap_or_default(),
                latency_ms: r.elapsed_ms as f32,
                tokens_per_second: 0.0,
            })
            .collect();

        let resp = BatchGenerateResponse {
            responses,
            total_latency_ms: total_latency_ms as f32,
            throughput_tokens_per_second: throughput_tokens_per_second as f32,
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
                        tokenizer
                            .encode(&owned, false, false)
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
                token_ids
                    .iter()
                    .map(|&id| {
                        tokenizer
                            .id_to_token(id)
                            .unwrap_or_else(|| format!("<id:{}>", id))
                    })
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
        let manager = self.model_manager.read().await;

        // Get actual model name from the loaded engine
        let model_name = match get_default_engine(&manager) {
            Some(engine_ref) => {
                let engine = engine_ref.read().await;
                match &*engine {
                    ModelEngine::PureCpp(e) => e.model_name.clone(),
                }
            }
            None => "no-model".to_string(),
        };

        let uptime = self.start_time.elapsed().as_secs() as i64;

        let resp = HealthResponse {
            status: if model_name != "no-model" { "ok" } else { "degraded" }.into(),
            version: self.version.clone(),
            model: model_name,
            uptime_seconds: uptime,
        };

        Ok(Response::new(resp))
    }

    async fn model_info(
        &self,
        _request: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        let manager = self.model_manager.read().await;

        let resp = match get_default_engine(&manager) {
            Some(engine_ref) => {
                let engine = engine_ref.read().await;
                match &*engine {
                    ModelEngine::PureCpp(e) => {
                        let tokenizer = manager.get_default_tokenizer().await;
                        let vocab_size = tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(0);
                        let info = e.info();

                        ModelInfoResponse {
                            model_name: info.name.clone(),
                            model_path: info.path.clone(),
                            max_context_length: 65536,
                            vocab_size: vocab_size as i32,
                            supports_streaming: true,
                            capabilities: vec![
                                "generate".into(),
                                "stream".into(),
                                "batch".into(),
                                "tokenize".into(),
                            ],
                        }
                    }
                }
            }
            None => ModelInfoResponse {
                model_name: "no-model".into(),
                model_path: "".into(),
                max_context_length: 0,
                vocab_size: 0,
                supports_streaming: false,
                capabilities: vec![],
            },
        };

        Ok(Response::new(resp))
    }
}
