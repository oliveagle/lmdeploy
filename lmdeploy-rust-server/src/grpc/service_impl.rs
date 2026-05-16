use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::{Response, Status};

use crate::cache::{TokenizeCache, compute_hash};
use crate::grpc::lmdeploy::v1::{
    BatchGenerateRequest, BatchGenerateResponse, GenerateRequest, GenerateResponse,
    GenerateStreamResponse, HealthRequest, HealthResponse, ModelInfoRequest, ModelInfoResponse,
    StreamChunk, TokenizeRequest, TokenizeResponse, lm_deploy_service_server::LmDeployService,
};

#[derive(Clone)]
pub struct LmDeployServiceImpl {
    pub version: String,
    pub tokenizer_cache: Arc<TokenizeCache>,
}

impl LmDeployServiceImpl {
    /// Mock tokenizer function - replace with real TurboMind tokenizer
    async fn mock_tokenize(text: &str) -> Vec<u32> {
        // Simple mock: each character becomes a token
        text.chars().map(|c| c as u32).collect()
    }
}

#[tonic::async_trait]
impl LmDeployService for LmDeployServiceImpl {
    async fn generate(
        &self,
        request: tonic::Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let req = request.into_inner();
        tracing::info!(prompt_len = req.prompt.len(), max_tokens = req.max_tokens, "Generate request");

        let start = std::time::Instant::now();
        let response = GenerateResponse {
            text: "Mock generated response - implement TurboMind FFI".into(),
            token_ids: vec![0; 10],
            prompt_tokens: req.prompt.len() as i32,
            completion_tokens: 10,
            finish_reason: 0.0,
            error: String::new(),
            latency_ms: start.elapsed().as_millis() as f32,
            tokens_per_second: 1000.0,
        };

        Ok(Response::new(response))
    }

    type GenerateStreamStream = Pin<Box<dyn Stream<Item = Result<GenerateStreamResponse, Status>> + Send>>;

    async fn generate_stream(
        &self,
        request: tonic::Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let req = request.into_inner();
        tracing::info!(prompt_len = req.prompt.len(), max_tokens = req.max_tokens, "Generate stream request");

        let (tx, mut rx) = mpsc::channel::<Result<GenerateStreamResponse, Status>>(32);

        tokio::spawn(async move {
            let tokens = ["Token 1", "Token 2", "Generated text"];

            for (i, token_text) in tokens.iter().enumerate() {
                let chunk = StreamChunk {
                    text: token_text.to_string(),
                    token_id: i as i32 + 100,
                    is_final: false,
                };

                let _ = tx.send(Ok(GenerateStreamResponse {
                    payload: Some(
                        crate::grpc::lmdeploy::v1::generate_stream_response::Payload::Chunk(chunk),
                    ),
                })).await;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }

            let final_response = GenerateResponse {
                text: tokens.join(" "),
                token_ids: vec![100, 101, 102],
                prompt_tokens: req.prompt.len() as i32,
                completion_tokens: 3,
                finish_reason: 0.0,
                error: String::new(),
                latency_ms: 150.0,
                tokens_per_second: 20.0,
            };

            let _ = tx.send(Ok(GenerateStreamResponse {
                payload: Some(
                    crate::grpc::lmdeploy::v1::generate_stream_response::Payload::Response(
                        final_response,
                    ),
                ),
            })).await;
        });

        let output_stream = async_stream::stream! {
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };

        Ok(Response::new(Box::pin(output_stream) as Self::GenerateStreamStream))
    }

    async fn generate_batch(
        &self,
        request: tonic::Request<BatchGenerateRequest>,
    ) -> Result<Response<BatchGenerateResponse>, Status> {
        let req = request.into_inner();
        let batch_size = req.batch_size.max(1) as usize;

        tracing::info!(request_count = req.requests.len(), batch_size, "Batch generate request");

        let start = std::time::Instant::now();
        let mut responses = Vec::with_capacity(req.requests.len());

        for batch_chunk in req.requests.chunks(batch_size) {
            for req_item in batch_chunk {
                let response = GenerateResponse {
                    text: format!("Batch response for: {}", req_item.prompt),
                    token_ids: vec![0; 5],
                    prompt_tokens: req_item.prompt.len() as i32,
                    completion_tokens: 5,
                    finish_reason: 0.0,
                    error: String::new(),
                    latency_ms: 100.0,
                    tokens_per_second: 50.0,
                };
                responses.push(response);
            }
        }

        let total_latency = start.elapsed().as_millis() as f32;
        let total_tokens: i32 = responses.iter().map(|r| r.completion_tokens).sum();
        let throughput = if total_latency > 0.0 {
            (total_tokens as f32) / (total_latency / 1000.0)
        } else {
            0.0
        };

        Ok(Response::new(BatchGenerateResponse {
            responses,
            total_latency_ms: total_latency,
            throughput_tokens_per_second: throughput,
        }))
    }

    async fn tokenize(
        &self,
        request: tonic::Request<TokenizeRequest>,
    ) -> Result<Response<TokenizeResponse>, Status> {
        let req = request.into_inner();
        tracing::info!(text_len = req.text.len(), "Tokenize request");

        let hash = compute_hash(&req.text);

        let token_ids = self
            .tokenizer_cache
            .get_or_tokenize(&req.text, |text| {
                let owned = text.to_string();
                async move { Ok(Self::mock_tokenize(&owned).await) }
            })
            .await
            .map_err(|e| Status::internal(format!("Tokenize error: {}", e)))?;

        let length = token_ids.len() as i32;
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();

        let response = TokenizeResponse {
            token_ids: token_ids_i32,
            tokens: vec![],
            length,
            hash,
        };

        Ok(Response::new(response))
    }

    async fn health(
        &self,
        _request: tonic::Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let _cache_metrics = self.tokenizer_cache.metrics().await;
        let _cache_hit_rate = self.tokenizer_cache.hit_rate().await;

        Ok(Response::new(HealthResponse {
            status: "ok".into(),
            version: self.version.clone(),
            model: "mock-model".into(),
            uptime_seconds: 0,
        }))
    }

    async fn model_info(
        &self,
        _request: tonic::Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        Ok(Response::new(ModelInfoResponse {
            model_name: "mock-model".into(),
            model_path: String::new(),
            max_context_length: 8192,
            vocab_size: 152064,
            supports_streaming: true,
            capabilities: vec!["generate".into(), "stream".into(), "batch".into()],
        }))
    }
}
