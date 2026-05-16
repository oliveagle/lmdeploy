use std::sync::Arc;

use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::cache::TokenizeCache;

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

        tokio::spawn(async move {
            for word in words {
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

        tokio::spawn(async move {
            while let Some(request_result) = stream.next().await {
                match request_result {
                    Ok(req) => {
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
                    Err(e) => {
                        tracing::error!("Error in bidirectional stream: {:?}", e);
                        let _ = tx
                            .send(Err(Status::invalid_argument(format!("Invalid request: {}", e))))
                            .await;
                        break;
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
        let token_ids: Vec<u32> = req.text.chars().map(|c| c as u32).collect();
        let tokens: Vec<String> = req.text.split_whitespace().map(String::from).collect();
        let length = token_ids.len();

        let resp = TokenizeResponse {
            token_ids,
            tokens,
            length: length as i32,
            hash,
        };

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