// src/protocol/common.mbt
// Common protocol definitions shared across all API types
// Reference: lmdeploy/serve/openai/protocol.py

/// Usage information for token counting
pub struct UsageInfo {
  prompt_tokens: Int
  total_tokens: Int
  completion_tokens: Int
}

pub fn UsageInfo::default() -> UsageInfo {
  { prompt_tokens: 0, total_tokens: 0, completion_tokens: 0 }
}

/// Embedding data for /v1/embeddings endpoint
pub struct EmbeddingData {
  object: String
  embedding: Array[Float]
  index: Int
}

/// Embeddings request
pub struct EmbeddingsRequest {
  model: String
  input: String
  input_array: Array[String]
  encoding_format: String
  dimensions: Int
}

pub fn EmbeddingsRequest::default() -> EmbeddingsRequest {
  {
    model: "",
    input: "",
    input_array: [],
    encoding_format: "base64",
    dimensions: 0,
  }
}

/// Embeddings response
pub struct EmbeddingsResponse {
  object: String
  data: Array[EmbeddingData]
  model: String
  usage: UsageInfo
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
  root: Option[String]
  parent: Option[String]
  permission: Array[ModelPermission]
}

pub fn ModelCard::default() -> ModelCard {
  {
    id: "",
    object: "model",
    created: 0L,
    owned_by: "lmdeploy",
    root: None,
    parent: None,
    permission: [],
  }
}

/// Model list response
pub struct ModelList {
  object: String
  data: Array[ModelCard]
}

pub fn ModelList::default() -> ModelList {
  { object: "list", data: [] }
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

/// Health check response
pub struct HealthResponse {
  status: String
  version: String
  model: String
  uptime_seconds: Int64
}

pub fn HealthResponse::default() -> HealthResponse {
  { status: "ok", version: "0.1.0", model: "", uptime_seconds: 0L }
}

/// Model info response
pub struct ModelInfoResponse {
  model_name: String
  model_path: String
  max_context_length: Int
  vocab_size: Int
  supports_streaming: Bool
  capabilities: Array[String]
}

pub fn ModelInfoResponse::default() -> ModelInfoResponse {
  {
    model_name: "",
    model_path: "",
    max_context_length: 4096,
    vocab_size: 151936,
    supports_streaming: true,
    capabilities: ["chat", "completion", "embedding", "streaming"],
  }
}

/// Error response
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

/// Build OpenAI error response JSON
pub fn build_error_json(code: Int, type_: String, message: String) -> String {
  let code_str = int_to_string(code)
  let msg_escaped = escape_json_string(message)
  let type_escaped = escape_json_string(type_)
  """
{"error": {"message": """" ++ msg_escaped ++ """", "type": """" ++ type_escaped ++ """", "param": null, "code": """ ++ code_str ++ """}}
"""
}

/// Build OpenAI chat completion response JSON
pub fn build_chat_completion_json(
  id: String,
  created: Int,
  model: String,
  message_role: String,
  message_content: String,
  usage: UsageInfo
) -> String {
  let created_str = int_to_string(created)
  let prompt_tokens_str = int_to_string(usage.prompt_tokens)
  let completion_tokens_str = int_to_string(usage.completion_tokens)
  let total_tokens_str = int_to_string(usage.total_tokens)

  """
{"id": """" ++ id ++ """", "object": "chat.completion", "created": """ ++ created_str ++ """, "model": """" ++ model ++ """", "choices": [{"index": 0, "message": {"role": """" ++ message_role ++ """", "content": """" ++ message_content ++ """", "tool_calls": null}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": """ ++ prompt_tokens_str ++ """, "completion_tokens": """ ++ completion_tokens_str ++ """, "total_tokens": """ ++ total_tokens_str ++ """}, "system_fingerprint": null}
"""
}

/// Escape special characters in JSON string
fn escape_json_string(s : String) -> String {
  let mut result = ""
  let bytes = s.to_utf8_bytes()
  for b in bytes {
    if b == ('"' as U8) { result = result ++ "\\\"" }
    else if b == ('\\' as U8) { result = result ++ "\\\\" }
    else if b == ('/' as U8) { result = result ++ "\\/" }
    else if b == ('\n' as U8) { result = result ++ "\\n" }
    else if b == ('\r' as U8) { result = result ++ "\\r" }
    else if b == ('\t' as U8) { result = result ++ "\\t" }
    else if b < 0x20 as U8 {
      // Control characters - escape as unicode
      result = result ++ "\\u00" ++ byte_to_hex(b)
    } else {
      result = result ++ String::from_byte(b)
    }
  }
  result
}

/// Convert byte to hex string
fn byte_to_hex(b : U8) -> String {
  let high = (b >> 4) & 0x0F
  let low = b & 0x0F
  digit_to_hex(high) ++ digit_to_hex(low)
}

/// Convert digit to hex char
fn digit_to_hex(d : U8) -> String {
  if d < 10 { String::from_byte(('0' as U8) + d) }
  else { String::from_byte(('a' as U8) + d - 10) }
}

/// Convert int to string (complete implementation)
fn int_to_string(i: Int) -> String {
  if i == 0 { "0" }
  else if i == 1 { "1" }
  else if i == 2 { "2" }
  else if i == 3 { "3" }
  else if i == 4 { "4" }
  else if i == 5 { "5" }
  else if i == 6 { "6" }
  else if i == 7 { "7" }
  else if i == 8 { "8" }
  else if i == 9 { "9" }
  else if i == 10 { "10" }
  else if i == 16 { "16" }
  else if i == 40 { "40" }
  else if i == 100 { "100" }
  else if i == 128 { "128" }
  else if i == 256 { "256" }
  else if i == 500 { "500" }
  else if i == 512 { "512" }
  else if i == 1000 { "1000" }
  else if i == 1024 { "1024" }
  else if i == 2048 { "2048" }
  else if i == 4096 { "4096" }
  else if i == 8192 { "8192" }
  else if i == 500 { "500" }
  else if i < 0 {
    "-" ++ uint_to_string(-i)
  } else {
    uint_to_string(i)
  }
}

/// Convert unsigned int to string
fn uint_to_string(n : Int) -> String {
  if n == 0 { "0" } else {
    let mut digits = []
    let mut num = n
    while num > 0 {
      let digit = num % 10
      digits = [digit_to_char(digit)] ++ digits
      num = num / 10
    }
    let mut result = ""
    for d in digits {
      result = result ++ d
    }
    result
  }
}

/// Convert digit to char
fn digit_to_char(d : Int) -> String {
  match d {
    0 => "0"
    1 => "1"
    2 => "2"
    3 => "3"
    4 => "4"
    5 => "5"
    6 => "6"
    7 => "7"
    8 => "8"
    _ => "9"
  }
}