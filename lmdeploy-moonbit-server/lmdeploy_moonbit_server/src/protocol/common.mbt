// src/protocol/common.mbt
// Common protocol definitions shared across all API types
// Reference: lmdeploy/serve/openai/protocol.py -> ErrorResponse, ModelCard, etc.

/// Error response matching Python ErrorResponse
pub struct ErrorResponse {
  message: String
  type_: String
  code: Int
  param: Option[String]
  object: String
}

pub fn ErrorResponse::default() -> ErrorResponse {
  { message: "", type_: "error", code: 500, param: None, object: "error" }
}

/// Model permission
pub struct ModelPermission {
  id: String
  object: String
  created: Int64
  allow_create_engine: Bool
  allow_sampling: Bool
  allow_logprobs: Bool
  allow_search_indices: Bool
  allow_view: Bool
  allow_fine_tuning: Bool
  organization: String
  group: Option[String]
  is_blocking: Bool
}

/// Model card for /v1/models endpoint
pub struct ModelCard {
  id: String
  object: String
  created: Int64
  owned_by: String
  root: Option<String>
  parent: Option[String>
  permission: Array[ModelPermission]
}

/// Model list response
pub struct ModelList {
  object: String
  data: Array[ModelCard]
}

pub fn ModelList::default() -> ModelList {
  { object: "list", data: [] }
}

/// Embedding data
pub struct EmbeddingData {
  object: String
  embedding: Array[Float]
  index: Int
}

/// Embeddings request
pub struct EmbeddingsRequest {
  model: String
  inputs: Array[String]
  encoding_format: String
  dimensions: Option[Int>
}

/// Embeddings response
pub struct EmbeddingsResponse {
  object: String
  data: Array[EmbeddingData]
  model: String
  usage: UsageInfo
}

/// Encode (tokenize) request
pub struct EncodeRequest {
  input: String
  do_preprocess: Bool
  add_bos: Bool
}

pub fn EncodeRequest::default() -> EncodeRequest {
  { input: "", do_preprocess: false, add_bos: true }
}

/// Encode response
pub struct EncodeResponse {
  input_ids: Array[Int]
  length: Int
}

/// Health check request
pub struct HealthRequest {}

/// Health check response
pub struct HealthResponse {
  status: String
  version: String
  model: String
  uptime_seconds: Int64
}

/// Model info request
pub struct ModelInfoRequest {}

/// Model info response
pub struct ModelInfoResponse {
  model_name: String
  model_path: String
  max_context_length: Int
  vocab_size: Int
  supports_streaming: Bool
  capabilities: Array[String]
}

/// Stream options
pub struct StreamOptions {
  include_usage: Bool
}

pub fn StreamOptions::default() -> StreamOptions {
  { include_usage: false }
}

/// Response format (text, json_object, json_schema, regex_schema)
pub struct ResponseFormat {
  type_: String
  json_schema: Option[JsonSchema>
  regex_schema: Option[String>
}

pub fn ResponseFormat::default() -> ResponseFormat {
  { type_: "text", json_schema: None, regex_schema: None }
}

/// JSON schema for structured output
pub struct JsonSchema {
  name: String
  description: Option<String>
  json_schema: Option[Map[String, String]>
}

/// Finish reason enum
pub enum FinishReason {
  Stop
  Length
  ToolCalls
  Error
  Abort
}

pub fn FinishReason::to_string(self : FinishReason) -> String {
  match self {
    Stop => "stop"
    Length => "length"
    ToolCalls => "tool_calls"
    Error => "error"
    Abort => "abort"
  }
}
