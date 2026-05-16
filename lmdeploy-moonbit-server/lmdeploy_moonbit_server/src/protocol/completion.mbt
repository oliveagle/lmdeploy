// src/protocol/completion.mbt
// Text completion protocol definitions (OpenAI-compatible)
// Reference: lmdeploy/serve/openai/protocol.py -> CompletionRequest, CompletionResponse

/// Completion request matching Python CompletionRequest
pub struct CompletionRequest {
  model: String
  prompt: String
  prompt_token_ids: Array[Int]
  suffix: String
  temperature: Float
  n: Int
  max_tokens: Int
  stop: Array[String]
  stream: Bool
  top_p: Float
  echo: Bool
  presence_penalty: Float
  frequency_penalty: Float
  seed: Int
  repetition_penalty: Float
  ignore_eos: Bool
  skip_special_tokens: Bool
  top_k: Int
  min_p: Float
  session_id: Int
  top_logprobs: Int
}

pub fn CompletionRequest::default(model: String, prompt: String) -> CompletionRequest {
  {
    model,
    prompt,
    prompt_token_ids: [],
    suffix: "",
    temperature: 0.7,
    n: 1,
    max_tokens: 16,
    stop: [],
    stream: false,
    top_p: 1.0,
    echo: false,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    seed: 0,
    repetition_penalty: 1.0,
    ignore_eos: false,
    skip_special_tokens: true,
    top_k: 40,
    min_p: 0.0,
    session_id: -1,
    top_logprobs: 0,
  }
}

/// Single completion response choice
pub struct CompletionResponseChoice {
  index: Int
  text: String
  logprobs: Option[LogProbs]
  gen_tokens: Array[Int]
  finish_reason: Option[String]
}

/// Log probabilities for completion
pub struct LogProbs {
  text_offset: Array[Int]
  token_logprobs: Array[Float]
  tokens: Array[String]
  top_logprobs: Array[Map[String, Float]]
}

/// Full completion response
pub struct CompletionResponse {
  id: String
  object: String
  created: Int64
  model: String
  choices: Array[CompletionResponseChoice]
  usage: UsageInfo
}

/// Completion stream response
pub struct CompletionStreamResponse {
  id: String
  object: String
  created: Int64
  model: String
  choices: Array[CompletionResponseStreamChoice]
  usage: Option[UsageInfo]
}

/// Single choice in completion stream
pub struct CompletionResponseStreamChoice {
  index: Int
  text: String
  logprobs: Option[LogProbs]
  gen_tokens: Array[Int]
  finish_reason: Option[String]
}

/// Generate a completion request ID
pub fn create_completion_id() -> String {
  "cmpl-"
}
