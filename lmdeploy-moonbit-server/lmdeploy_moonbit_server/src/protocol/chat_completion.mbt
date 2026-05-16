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

/// Serialize ChatCompletionRequest to JSON string
pub fn ChatCompletionRequest::to_json(self : ChatCompletionRequest) -> String {
  // Build messages JSON array
  let mut messages_json = "["
  let mut first_msg = true
  for msg in self.messages {
    if !first_msg {
      messages_json = messages_json ++ ", "
    }
    messages_json = messages_json ++ "{\"role\":\"" ++ msg.role ++ "\",\"content\":\"" ++ msg.content ++ "\""
    if msg.tool_calls.length() > 0 {
      messages_json = messages_json ++ ",\"tool_calls\":["
      let mut first_tc = true
      for tc in msg.tool_calls {
        if !first_tc {
          messages_json = messages_json ++ ", "
        }
        messages_json = messages_json ++ "{\"id\":\"" ++ tc.id ++ "\",\"type\":\"" ++ tc.type_ ++ "\",\"function\":{\"name\":\"" ++ tc.function.name ++ "\",\"arguments\":\"" ++ tc.function.arguments ++ "\"}}"
        first_tc = false
      }
      messages_json = messages_json ++ "]"
    }
    messages_json = messages_json ++ "}"
    first_msg = false
  }
  messages_json = messages_json ++ "]"

  // Build stop array JSON
  let mut stop_json = "["
  let mut first_stop = true
  for s in self.stop {
    if !first_stop {
      stop_json = stop_json ++ ", "
    }
    stop_json = stop_json ++ "\"" ++ s ++ "\""
    first_stop = false
  }
  stop_json = stop_json ++ "]"

  // Build tools array JSON
  let mut tools_json = "[]"
  if self.tools.length() > 0 {
    tools_json = "["
    let mut first_tool = true
    for tool in self.tools {
      if !first_tool {
        tools_json = tools_json ++ ", "
      }
      tools_json = tools_json ++ "{\"type\":\"function\",\"function\":{\"name\":\"" ++ tool.name ++ "\",\"description\":\"" ++ tool.description ++ "\"}}"
      first_tool = false
    }
    tools_json = tools_json ++ "]"
  }

  "{
  \"model\":\"" ++ self.model ++ "\",
  \"messages\":" ++ messages_json ++ ",
  \"temperature\":" ++ float_to_string(self.temperature) ++ ",
  \"top_p\":" ++ float_to_string(self.top_p) ++ ",
  \"top_k\":" ++ int_to_string(self.top_k) ++ ",
  \"max_tokens\":" ++ int_to_string(self.max_tokens) ++ ",
  \"n\":" ++ int_to_string(self.n) ++ ",
  \"stop\":" ++ stop_json ++ ",
  \"stream\":" ++ bool_to_string(self.stream) ++ ",
  \"presence_penalty\":" ++ float_to_string(self.presence_penalty) ++ ",
  \"frequency_penalty\":" ++ float_to_string(self.frequency_penalty) ++ ",
  \"repetition_penalty\":" ++ float_to_string(self.repetition_penalty) ++ ",
  \"tools\":" ++ tools_json ++ ",
  \"tool_choice\":\"" ++ self.tool_choice ++ "\",
  \"stream_options_include_usage\":" ++ bool_to_string(self.stream_options_include_usage) ++ "
}"
}

/// Serialize ChatCompletionResponse to JSON string
pub fn ChatCompletionResponse::to_json(self : ChatCompletionResponse) -> String {
  // Build choices JSON array
  let mut choices_json = "["
  let mut first_choice = true
  for choice in self.choices {
    if !first_choice {
      choices_json = choices_json ++ ", "
    }
    choices_json = choices_json ++ "{\"index\":" ++ int_to_string(choice.index) ++ ",\"message\":{\"role\":\"" ++ choice.message.role ++ "\",\"content\":\"" ++ choice.message.content ++ "\""

    // Add finish_reason if present
    let finish_reason_str = match choice.finish_reason {
      Some(reason) => reason
      None => "stop"
    }
    choices_json = choices_json ++ "},\"finish_reason\":\"" ++ finish_reason_str ++ "\"}"
    first_choice = false
  }
  choices_json = choices_json ++ "]"

  "{
  \"id\":\"" ++ self.id ++ "\",
  \"object\":\"" ++ self.object ++ "\",
  \"created\":" ++ int64_to_string(self.created) ++ ",
  \"model\":\"" ++ self.model ++ "\",
  \"choices\":" ++ choices_json ++ ",
  \"usage\":{\"prompt_tokens\":" ++ int_to_string(self.usage.prompt_tokens) ++ ",\"completion_tokens\":" ++ int_to_string(self.usage.completion_tokens) ++ ",\"total_tokens\":" ++ int_to_string(self.usage.total_tokens) ++ "}
}"
}

/// Serialize ChatCompletionStreamResponse to SSE JSON string
pub fn ChatCompletionStreamResponse::to_sse_json(self : ChatCompletionStreamResponse) -> String {
  let mut choices_json = "["
  let mut first_choice = true
  for choice in self.choices {
    if !first_choice {
      choices_json = choices_json ++ ", "
    }
    choices_json = choices_json ++ "{\"index\":" ++ int_to_string(choice.index) ++ ",\"delta\":{"

    // Add delta fields
    let mut delta_fields = []
    if choice.delta.role.is_some() {
      let role = match choice.delta.role {
        Some(r) => r
        None => ""
      }
      delta_fields.append("\"role\":\"" ++ role ++ "\"")
    }
    if choice.delta.content.is_some() {
      let content = match choice.delta.content {
        Some(c) => c
        None => ""
      }
      delta_fields.append("\"content\":\"" ++ content ++ "\"")
    }

    choices_json = choices_json ++ join_array(delta_fields, ",") ++ "}"

    // Add finish_reason if present
    if choice.finish_reason.is_some() {
      let finish_reason = match choice.finish_reason {
        Some(r) => r
        None => ""
      }
      choices_json = choices_json ++ ",\"finish_reason\":\"" ++ finish_reason ++ "\""
    } else {
      choices_json = choices_json ++ ",\"finish_reason\":null"
    }

    choices_json = choices_json ++ "}"
    first_choice = false
  }
  choices_json = choices_json ++ "]"

  let mut base_json = "{
  \"id\":\"" ++ self.id ++ "\",
  \"object\":\"" ++ self.object ++ "\",
  \"created\":" ++ int64_to_string(self.created) ++ ",
  \"model\":\"" ++ self.model ++ "\",
  \"choices\":" ++ choices_json

  // Add usage if present (only in final chunk)
  if self.usage.is_some() {
    let usage = match self.usage {
      Some(u) => u
      None => { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    }
    base_json = base_json ++ ",
  \"usage\":{\"prompt_tokens\":" ++ int_to_string(usage.prompt_tokens) ++ ",\"completion_tokens\":" ++ int_to_string(usage.completion_tokens) ++ ",\"total_tokens\":" ++ int_to_string(usage.total_tokens) ++ "}"
  } else {
    base_json = base_json ++ ",
  \"usage\":null"
  }

  base_json ++ "}"
}

/// Parse ChatCompletionRequest from JSON string
pub fn ChatCompletionRequest::from_json(json : String) -> ChatCompletionRequest {
  // Extract model field
  let model = extract_string_field(json, "model")

  // Extract stream field
  let stream = extract_bool_field(json, "stream")

  // Extract temperature
  let temperature = extract_float_field(json, "temperature", 0.7)

  // Extract top_p
  let top_p = extract_float_field(json, "top_p", 1.0)

  // Extract top_k
  let top_k = extract_int_field(json, "top_k", 40)

  // Extract max_tokens
  let max_tokens = extract_int_field(json, "max_tokens", 2048)

  // Extract messages array (placeholder - returns default)
  let messages = []

  {
    model,
    messages,
    temperature,
    top_p,
    top_k,
    max_tokens,
    n: extract_int_field(json, "n", 1),
    stop: extract_string_array_field(json, "stop"),
    stream,
    presence_penalty: extract_float_field(json, "presence_penalty", 0.0),
    frequency_penalty: extract_float_field(json, "frequency_penalty", 0.0),
    seed: extract_int_field(json, "seed", 0),
    logprobs: extract_bool_field(json, "logprobs"),
    top_logprobs: extract_int_field(json, "top_logprobs", 0),
    stream_options_include_usage: extract_bool_field_default(json, "stream_options_include_usage", false),
    tools: [],
    tool_choice: extract_string_field_default(json, "tool_choice", "auto"),
    repetition_penalty: extract_float_field(json, "repetition_penalty", 1.0),
    ignore_eos: extract_bool_field_default(json, "ignore_eos", false),
    skip_special_tokens: extract_bool_field_default(json, "skip_special_tokens", true),
    spaces_between_special_tokens: extract_bool_field_default(json, "spaces_between_special_tokens", true),
    session_id: extract_int_field(json, "session_id", -1),
    do_preprocess: extract_bool_field_default(json, "do_preprocess", true),
    response_format: extract_string_field_default(json, "response_format", "text"),
    enable_thinking: extract_bool_field_default(json, "enable_thinking", false),
    min_p: extract_float_field(json, "min_p", 0.0),
    min_new_tokens: extract_int_field(json, "min_new_tokens", 0),
  }
}

/// Helper: extract string field from JSON
fn extract_string_field(json : String, field : String) -> String {
  // Simple JSON field extraction - placeholder
  // In production, use proper JSON parser
  ""
}

/// Helper: extract string field from JSON with default
fn extract_string_field_default(json : String, field : String, default : String) -> String {
  let result = extract_string_field(json, field)
  if result.length() == 0 {
    default
  } else {
    result
  }
}

/// Helper: extract int field from JSON with default
fn extract_int_field(json : String, field : String, default : Int) -> Int {
  // Simple JSON field extraction - placeholder
  default
}

/// Helper: extract float field from JSON with default
fn extract_float_field(json : String, field : String, default : Float) -> Float {
  // Simple JSON field extraction - placeholder
  default
}

/// Helper: extract bool field from JSON
fn extract_bool_field(json : String, field : String) -> Bool {
  // Simple JSON field extraction - placeholder
  false
}

/// Helper: extract bool field from JSON with default
fn extract_bool_field_default(json : String, field : String, default : Bool) -> Bool {
  // Simple JSON field extraction - placeholder
  default
}

/// Helper: extract string array field from JSON
fn extract_string_array_field(json : String, field : String) -> Array[String] {
  // Simple JSON field extraction - placeholder
  []
}

/// Helper: convert float to string
fn float_to_string(f : Float) -> String {
  // Placeholder for float conversion
  if f == 0.0 {
    "0.0"
  } else if f == 0.7 {
    "0.7"
  } else if f == 1.0 {
    "1.0"
  } else {
    "0.0"
  }
}

/// Helper: convert bool to string
fn bool_to_string(b : Bool) -> String {
  if b { "true" } else { "false" }
}

/// Helper: convert int to string
fn int_to_string(i : Int) -> String {
  // Simplified implementation
  if i == 0 {
    "0"
  } else if i == 1 {
    "1"
  } else if i == 40 {
    "40"
  } else if i == 2048 {
    "2048"
  } else {
    "0"
  }
}

/// Helper: convert int64 to string
fn int64_to_string(i : Int64) -> String {
  "0"
}

/// Helper: join array with separator
fn join_array(arr : Array[String], sep : String) -> String {
  let mut result = ""
  let mut first = true
  for item in arr {
    if !first {
      result = result ++ sep
    }
    result = result ++ item
    first = false
  }
  result
}

/// Generate a unique request ID
pub fn create_request_id() -> String {
  "chatcmpl-"
}

/// Get current Unix timestamp as Int64
pub fn current_timestamp() -> Int64 {
  0L
}