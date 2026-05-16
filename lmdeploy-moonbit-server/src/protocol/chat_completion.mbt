// src/protocol/chat_completion.mbt
// Chat completion protocol definitions (OpenAI-compatible)
// Reference: lmdeploy/serve/openai/protocol.py

/// Chat completion request matching Python ChatCompletionRequest
pub struct ChatCompletionRequest {
  model: String
  messages: Array[ChatMessage]
  temperature: Float
  top_p: Float
  top_k: Int
  max_tokens: Int
  n: Int
  stop: Array[String]
  stream: Bool
  presence_penalty: Float
  frequency_penalty: Float
  seed: Int
  logprobs: Bool
  top_logprobs: Int
  stream_options_include_usage: Bool
  tools: Array[Tool]
  tool_choice: String
  repetition_penalty: Float
  ignore_eos: Bool
  skip_special_tokens: Bool
  spaces_between_special_tokens: Bool
  session_id: Int
  do_preprocess: Bool
  response_format: String
  enable_thinking: Bool
  min_p: Float
  min_new_tokens: Int
}

pub fn ChatCompletionRequest::default(model: String, messages: Array[ChatMessage]) -> ChatCompletionRequest {
  {
    model,
    messages,
    temperature: 0.7,
    top_p: 1.0,
    top_k: 40,
    max_tokens: 2048,
    n: 1,
    stop: [],
    stream: false,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    seed: 0,
    logprobs: false,
    top_logprobs: 0,
    stream_options_include_usage: false,
    tools: [],
    tool_choice: "auto",
    repetition_penalty: 1.0,
    ignore_eos: false,
    skip_special_tokens: true,
    spaces_between_special_tokens: true,
    session_id: -1,
    do_preprocess: true,
    response_format: "text",
    enable_thinking: false,
    min_p: 0.0,
    min_new_tokens: 0,
  }
}

/// Chat message (user/assistant/system/tool)
pub struct ChatMessage {
  role: String
  content: String
  reasoning_content: String
  tool_calls: Array[ToolCall]
}

pub fn ChatMessage::default() -> ChatMessage {
  { role: "", content: "", reasoning_content: "", tool_calls: [] }
}

/// Tool definition with function schema
pub struct Tool {
  name: String
  description: String
  parameters: Map[String, String]
}

/// Tool call in response
pub struct ToolCall {
  id: String
  type_: String
  function: FunctionCall
}

/// Function call with name and JSON arguments
pub struct FunctionCall {
  name: String
  arguments: String
}

/// Chat completion response
pub struct ChatCompletionResponse {
  id: String
  object: String
  created: Int64
  model: String
  choices: Array[ChatCompletionResponseChoice]
  usage: UsageInfo
}

/// Single choice in chat completion response
pub struct ChatCompletionResponseChoice {
  index: Int
  message: ChatMessage
  logprobs: Option[ChoiceLogprobs]
  finish_reason: Option[String]
}

/// Logprobs for a choice
pub struct ChoiceLogprobs {
  content: Array[ChatCompletionTokenLogprob]
}

/// Token logprob with top-k alternatives
pub struct ChatCompletionTokenLogprob {
  token: String
  token_bytes: Array[Byte]
  logprob: Float
  top_logprobs: Array[TopLogprob]
}

/// Top log probability for a token
pub struct TopLogprob {
  token: String
  token_bytes: Array[Byte]
  logprob: Float
}

/// Chat completion stream response (SSE-like)
pub struct ChatCompletionStreamResponse {
  id: String
  object: String
  created: Int64
  model: String
  choices: Array[ChatCompletionResponseStreamChoice]
  usage: Option[UsageInfo]
}

/// Single choice in stream response
pub struct ChatCompletionResponseStreamChoice {
  index: Int
  delta: DeltaMessage
  logprobs: Option[ChoiceLogprobs]
  finish_reason: Option[String]
}

/// Delta message in streaming response
pub struct DeltaMessage {
  role: Option[String]
  content: Option[String]
  reasoning_content: Option[String]
  tool_calls: Array[DeltaToolCall]
}

/// Tool call delta in streaming
pub struct DeltaToolCall {
  id: Option[String]
  index: Int
  type_: Option[String]
  function: Option[DeltaFunctionCall]
}

/// Function call delta in streaming
pub struct DeltaFunctionCall {
  name: Option[String]
  arguments: Option[String]
}

/// Batch chat completion request
pub struct BatchChatCompletionRequest {
  requests: Array[ChatCompletionRequest]
  batch_size: Int
}

/// Batch chat completion response
pub struct BatchChatCompletionResponse {
  responses: Array[ChatCompletionResponse]
  total_latency_ms: Float
  throughput_tokens_per_second: Float
}

/// Usage information for token counting
pub struct UsageInfo {
  prompt_tokens: Int
  total_tokens: Int
  completion_tokens: Int
}

pub fn UsageInfo::default() -> UsageInfo {
  { prompt_tokens: 0, total_tokens: 0, completion_tokens: 0 }
}

/// JSON value type for flexible parameter handling
pub enum JsonValue {
  JsonNull
  JsonBool(Bool)
  JsonNumber(Float)
  JsonString(String)
  JsonArray(Array[JsonValue])
  JsonObject(Map[String, JsonValue])
}

/// Generate a unique request ID
pub fn create_request_id() -> String {
  "chatcmpl-"
}

/// Get current Unix timestamp as Int64
pub fn current_timestamp() -> Int64 {
  0L
}