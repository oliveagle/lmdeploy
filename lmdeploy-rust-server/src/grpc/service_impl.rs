use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::cache::TokenizeCache;
use crate::error::AppError;
use crate::metrics::{EngineEventType, StreamRequestMetrics};
use crate::model::ModelManager;
use crate::model::{BatchItem, BatchResult, GenerationParams, ModelEngine};
use crate::turbomind_c::CompiledGrammar;

use super::lmdeploy::v1::{
    generate_stream_response, lm_deploy_service_server::LmDeployService, BatchGenerateRequest,
    BatchGenerateResponse, GenerateRequest, GenerateResponse, GenerateStreamResponse,
    HealthRequest, HealthResponse, LogprobEntry, ModelInfoRequest, ModelInfoResponse, StreamChunk,
    TokenizeRequest, TokenizeResponse, TopLogprobEntry, EngineEventType as ProtoEngineEventType,
    StreamMetrics, EngineEvent,
};

/// Convert guided decoding fields from gRPC request to a CompiledGrammar.
fn build_grammar_from_grpc(req: &GenerateRequest) -> Option<Arc<CompiledGrammar>> {
    if !req.json_schema.is_empty() {
        match CompiledGrammar::from_json_schema(&req.json_schema) {
            Ok(grammar) => Some(Arc::new(grammar)),
            Err(e) => {
                tracing::warn!("Failed to compile JSON schema from gRPC request: {}", e);
                None
            }
        }
    } else if !req.ebnf_grammar.is_empty() {
        match CompiledGrammar::from_ebnf(&req.ebnf_grammar) {
            Ok(grammar) => Some(Arc::new(grammar)),
            Err(e) => {
                tracing::warn!("Failed to compile EBNF grammar from gRPC request: {}", e);
                None
            }
        }
    } else if !req.regex_pattern.is_empty() {
        match CompiledGrammar::from_regex(&req.regex_pattern) {
            Ok(grammar) => Some(Arc::new(grammar)),
            Err(e) => {
                tracing::warn!("Failed to compile regex from gRPC request: {}", e);
                None
            }
        }
    } else if req.builtin_json_grammar {
        Some(Arc::new(CompiledGrammar::builtin_json()))
    } else {
        None
    }
}

/// Helper to get the default engine from model manager
/// This should be called with &model_manager (not a read guard)
fn get_default_engine(
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Option<std::sync::Arc<tokio::sync::RwLock<ModelEngine>>> {
    // Note: This returns a clone of the Arc, not a borrow to the internal model
    model_manager.blocking_read().get_model(None)
}

/// Convert internal EngineEventType to proto-generated EngineEventType i32 value
fn event_type_to_proto(t: EngineEventType) -> i32 {
    match t {
        EngineEventType::Queued => ProtoEngineEventType::EngineEventQueued as i32,
        EngineEventType::Scheduled => ProtoEngineEventType::EngineEventScheduled as i32,
        EngineEventType::Preempted => ProtoEngineEventType::EngineEventPreempted as i32,
    }
}

/// Build optional proto StreamMetrics from a StreamRequestMetrics
fn build_stream_metrics(metrics: &StreamRequestMetrics) -> Option<StreamMetrics> {
    Some(StreamMetrics {
        token_timestamp_secs: metrics.token_timestamp_secs,
        engine_events: metrics
            .engine_events
            .iter()
            .map(|e| EngineEvent {
                event_type: event_type_to_proto(e.event_type),
                timestamp_secs: e.timestamp_secs,
            })
            .collect(),
    })
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

        // Check if logprobs are requested
        let _need_logprobs = req.logprobs || req.top_logprobs > 0;
        let _top_logprobs = if req.top_logprobs > 0 {
            Some(req.top_logprobs as u32)
        } else {
            None
        };

        // Get the engine from model manager
        let engine_ref = match get_default_engine(&self.model_manager) {
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
                    logprobs: vec![],
                };
                return Ok(Response::new(resp));
            }
        };

        let engine = engine_ref.read().await;
        let eng = &*engine;

        // Build grammar from guided decoding fields
        let grammar = build_grammar_from_grpc(&req);

        // Build GenerationParams with logprobs and grammar support
        let need_logprobs = req.logprobs || req.top_logprobs > 0;
        let mut params = GenerationParams::from_grpc_request_with_logprobs(
            if req.max_tokens > 0 { Some(req.max_tokens as usize) } else { None },
            if req.temperature > 0.0 { Some(req.temperature) } else { None },
            if req.top_p > 0.0 { Some(req.top_p) } else { None },
            if req.top_k > 0 { Some(req.top_k) } else { None },
            if req.repetition_penalty > 0 { Some(req.repetition_penalty as f32) } else { None },
            if req.seed > 0 { Some(req.seed as u64) } else { None },
            if req.logprobs { Some(true) } else { None },
            if req.top_logprobs > 0 { Some(req.top_logprobs as u32) } else { None },
        );

        // Apply grammar constraint if provided
        params.grammar = grammar;

        // Run inference with or without logprobs
        let (text, num_tokens, elapsed_ms, logprobs) = if need_logprobs {
            let (t, nt, el, lp) = eng.generate_with_logprobs(&req.prompt, params).await;
            (t, nt, el, lp)
        } else {
            let t = eng.generate(&req.prompt, params).await;
            (t, 0, 0.0, None)
        };

        // Recalculate elapsed_ms for non-logprobs path
        let elapsed = if need_logprobs { elapsed_ms } else {
            // Use a simple estimate if we didn't get metrics
            let _ = num_tokens; // suppress unused warning
            0.0
        };

        // Calculate tokens per second
        let tokens_per_second = if elapsed > 0.0 && num_tokens > 0 {
            (num_tokens as f64 / elapsed) * 1000.0
        } else {
            0.0
        };

        // Count prompt tokens if we have a tokenizer
        let prompt_tokens = self
            .model_manager
            .read()
            .await
            .get_default_tokenizer()
            .await
            .and_then(|t| t.encode(&req.prompt, false, false).ok())
            .map(|ids| ids.len())
            .unwrap_or(0);

        // Convert TokenLogprob to gRPC LogprobEntry
        let logprobs_entries = logprobs.map(|lp| {
            lp.iter()
                .map(|t| LogprobEntry {
                    token_id: 0, // Will be filled from token_ids if needed
                    token: t.token.clone(),
                    logprob: t.logprob as f32,
                    top_logprobs: t
                        .top_logprobs
                        .iter()
                        .map(|tp| TopLogprobEntry {
                            token_id: 0,
                            token: tp.token.clone(),
                            logprob: tp.logprob as f32,
                        })
                        .collect(),
                })
                .collect()
        }).unwrap_or_default();

        let is_empty = text.is_empty();
        let resp = GenerateResponse {
            text,
            token_ids: vec![],
            prompt_tokens: prompt_tokens as i32,
            completion_tokens: num_tokens as i32,
            finish_reason: if !is_empty { 0.0 } else { 2.0 },
            error: String::new(),
            latency_ms: elapsed as f32,
            tokens_per_second: tokens_per_second as f32,
            logprobs: logprobs_entries,
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

        // Pre-tokenize outside of spawn_blocking for lower TTFT
        // This moves tokenization off the critical path
        let pre_tokenized_ids: Option<Vec<u32>> = {
            let mm = self.model_manager.read().await;
            mm.get_default_tokenizer()
                .await
                .and_then(|t| t.encode(&prompt, false, false).ok())
        };

        tokio::spawn(async move {
            let first_token_start = Instant::now();
            let mut first_token_recorded = false;

            // Initialize streaming metrics with QUEUED event
            let mut metrics = StreamRequestMetrics::new();
            metrics.record_event(EngineEventType::Queued);

            // Get the engine
            let Some(engine_ref) = get_default_engine(&manager) else {
                let _ = tx.send(Ok(GenerateStreamResponse {
                    payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                        text: "[ERROR: No model loaded]".to_string(),
                        token_id: 0,
                        is_final: true,
                        metrics: build_stream_metrics(&metrics),
                    })),
                })).await;
                return;
            };

            // Lock the engine for the duration of streaming
            let engine_guard = engine_ref.read().await;
            let engine = &*engine_guard;

            // Record SCHEDULED event before starting inference
            metrics.record_event(EngineEventType::Scheduled);

            // Use the engine's streaming method with pre-tokenized input if available
            let mut stream = if let Some(ids) = pre_tokenized_ids {
                engine.generate_stream_with_ids(&prompt, ids, params).await
            } else {
                engine.generate_stream(&prompt, params).await
            };

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

                // Update token timestamp in metrics
                metrics.mark_token_generated();

                let chunk = GenerateStreamResponse {
                    payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                        text: token,
                        token_id: 0,
                        is_final: false,
                        metrics: build_stream_metrics(&metrics),
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
                    metrics: build_stream_metrics(&metrics),
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

                                // Initialize metrics for this request
                                let mut request_metrics = StreamRequestMetrics::new();
                                request_metrics.record_event(EngineEventType::Queued);

                                // Get the engine for this request
                                let Some(engine_ref) = get_default_engine(&manager) else {
                                    let _ = tx.send(Ok(GenerateStreamResponse {
                                        payload: Some(generate_stream_response::Payload::Chunk(StreamChunk {
                                            text: "[ERROR: No model loaded]".to_string(),
                                            token_id: 0,
                                            is_final: true,
                                            metrics: build_stream_metrics(&request_metrics),
                                        })),
                                    })).await;
                                    continue;
                                };
                                let engine_guard = engine_ref.read().await;
                                let engine = &*engine_guard;

                                // Record SCHEDULED event
                                request_metrics.record_event(EngineEventType::Scheduled);

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
                                    request_metrics.mark_token_generated();

                                    let chunk = GenerateStreamResponse {
                                        payload: Some(
                                            generate_stream_response::Payload::Chunk(StreamChunk {
                                                text: token,
                                                token_id: 0,
                                                is_final: false,
                                                metrics: build_stream_metrics(&request_metrics),
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
                                            metrics: build_stream_metrics(&request_metrics),
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

        let engine_ref = match get_default_engine(&self.model_manager) {
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
            .map(|(idx, gen_req)| {
                let grammar = build_grammar_from_grpc(&gen_req);
                let need_logprobs = gen_req.logprobs || gen_req.top_logprobs > 0;
                let mut params = GenerationParams::from_grpc_request_with_logprobs(
                    if gen_req.max_tokens > 0 { Some(gen_req.max_tokens as usize) } else { None },
                    if gen_req.temperature > 0.0 { Some(gen_req.temperature) } else { None },
                    if gen_req.top_p > 0.0 { Some(gen_req.top_p) } else { None },
                    if gen_req.top_k > 0 { Some(gen_req.top_k) } else { None },
                    if gen_req.repetition_penalty > 0 { Some(gen_req.repetition_penalty as f32) } else { None },
                    if gen_req.seed > 0 { Some(gen_req.seed as u64) } else { None },
                    if gen_req.logprobs { Some(true) } else { None },
                    if gen_req.top_logprobs > 0 { Some(gen_req.top_logprobs as u32) } else { None },
                );
                params.grammar = grammar;
                BatchItem {
                    request_id: idx as u64,
                    prompt: gen_req.prompt,
                    params,
                    need_logprobs,
                }
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
            .map(|r| {
                let logprobs = r.logprobs.unwrap_or_default().into_iter().map(|lp| LogprobEntry {
                    token_id: lp.token_id,
                    token: lp.token,
                    logprob: lp.logprob as f32,
                    top_logprobs: lp.top_logprobs.into_iter().map(|tlp| TopLogprobEntry {
                        token_id: tlp.token_id,
                        token: tlp.token,
                        logprob: tlp.logprob as f32,
                    }).collect(),
                }).collect();
                GenerateResponse {
                    text: r.text,
                    token_ids: vec![],
                    prompt_tokens: 0,
                    completion_tokens: r.num_tokens as i32,
                    finish_reason: if r.error.is_none() { 0.0 } else { 2.0 },
                    error: r.error.unwrap_or_default(),
                    latency_ms: r.elapsed_ms as f32,
                    tokens_per_second: 0.0,
                    logprobs,
                }
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
        // Get actual model name from the loaded engine
        let model_name = match get_default_engine(&self.model_manager) {
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
        let resp = match get_default_engine(&self.model_manager) {
            Some(engine_ref) => {
                let engine = engine_ref.read().await;
                match &*engine {
                    ModelEngine::PureCpp(e) => {
                        let tokenizer = self.model_manager.read().await.get_default_tokenizer().await;
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
