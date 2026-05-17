// src/protocol/chat_completion.mbt
// Chat completion protocol definitions (OpenAI-compatible)
// Reference: lmdeploy/serve/openai/protocol.py
use protocol.json_util

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
  let model = parse_json_string_field(json, "model", "")

  // Extract messages array
  let messages = parse_chat_messages(json)

  // Extract stream field
  let stream = parse_json_bool(json, "stream", false)

  {
    model,
    messages,
    temperature: parse_json_float(json, "temperature", 0.7),
    top_p: parse_json_float(json, "top_p", 1.0),
    top_k: parse_json_int(json, "top_k", 40),
    max_tokens: parse_json_int(json, "max_tokens", 2048),
    n: parse_json_int(json, "n", 1),
    stop: parse_json_string_array(json, "stop"),
    stream,
    presence_penalty: parse_json_float(json, "presence_penalty", 0.0),
    frequency_penalty: parse_json_float(json, "frequency_penalty", 0.0),
    seed: parse_json_int(json, "seed", 0),
    logprobs: parse_json_bool(json, "logprobs", false),
    top_logprobs: parse_json_int(json, "top_logprobs", 0),
    stream_options_include_usage: parse_json_bool(json, "stream_options_include_usage", false),
    tools: [],
    tool_choice: parse_json_string_field(json, "tool_choice", "auto"),
    repetition_penalty: parse_json_float(json, "repetition_penalty", 1.0),
    ignore_eos: parse_json_bool(json, "ignore_eos", false),
    skip_special_tokens: parse_json_bool(json, "skip_special_tokens", true),
    spaces_between_special_tokens: parse_json_bool(json, "spaces_between_special_tokens", true),
    session_id: parse_json_int(json, "session_id", -1),
    do_preprocess: parse_json_bool(json, "do_preprocess", true),
    response_format: parse_json_string_field(json, "response_format", "text"),
    enable_thinking: parse_json_bool(json, "enable_thinking", false),
    min_p: parse_json_float(json, "min_p", 0.0),
    min_new_tokens: parse_json_int(json, "min_new_tokens", 0),
  }
}

/// Parse chat messages array from JSON
fn parse_chat_messages(json : String) -> Array[ChatMessage] {
  let pattern = "\"messages\":["
  let idx = json.find(pattern)
  if idx < 0 {
    []
  } else {
    let rest = json.substring(idx + pattern.length(), json.length())
    let mut result = []
    let mut i = 0
    let mut brace_depth = 0
    let mut in_string = false
    let mut curr_msg = ""
    let mut started = false

    while i < rest.length() {
      let ch = rest.substring(i, i + 1)
      let b = rest.substring(i, i + 1).to_utf8_bytes()[0]

      if b == ('"' as U8) {
        // Track if we're entering or leaving a string
        // Look at previous char to determine if this is an escape
        let prev_escaped = if i > 0 {
          let prev_idx = i - 1
          while prev_idx >= 0 {
            let prev_b = rest.substring(prev_idx, i).to_utf8_bytes()[0]
            if prev_b != ('\\' as U8) { break }
            prev_idx = prev_idx - 1
          }
          prev_idx >= 0 && rest.substring(prev_idx, prev_idx + 1).to_utf8_bytes()[0] == ('\\' as U8)
        } else { false }

        if !prev_escaped {
          in_string = !in_string
        }
      }

      if !in_string {
        if ch == "{" {
          brace_depth = brace_depth + 1
          started = true
        } else if ch == "}" {
          brace_depth = brace_depth - 1
          if brace_depth == 0 && started {
            curr_msg = curr_msg ++ ch
            let msg = parse_single_message(curr_msg)
            if msg.role != "" {
              result = result.append([msg])
            }
            curr_msg = ""
            started = false
          }
        }
      }

      if started && brace_depth > 0 {
        curr_msg = curr_msg ++ ch
      }
      i = i + 1
    }
    result
  }
}

/// Parse a single chat message from JSON object
fn parse_single_message(msg_json : String) -> ChatMessage {
  let role = parse_json_string_field(msg_json, "role", "")
  let content = parse_json_string_field(msg_json, "content", "")
  let reasoning_content = parse_json_string_field(msg_json, "reasoning_content", "")

  // Parse tool_calls if present
  let mut tool_calls = []
  let tool_pattern = "\"tool_calls\":["
  let tool_idx = msg_json.find(tool_pattern)
  if tool_idx >= 0 {
    let rest = msg_json.substring(tool_idx + tool_pattern.length(), msg_json.length())
    // Simple extraction of tool_calls
    let mut i = 0
    let mut depth = 0
    let mut in_obj = false
    let mut current_tc = ""
    while i < rest.length() {
      let ch = rest.substring(i, i + 1)
      if ch == "{" { depth = depth + 1; in_obj = true }
      else if ch == "}" {
        depth = depth - 1
        if depth == 0 && in_obj {
          current_tc = current_tc ++ "}"
          let tc = parse_single_tool_call(current_tc)
          if tc.id != "" {
            tool_calls = tool_calls.append([tc])
          }
          current_tc = ""
          in_obj = false
        } else {
          current_tc = current_tc ++ ch
        }
      } else if in_obj {
        current_tc = current_tc ++ ch
      }
      if ch == "]" && depth == 0 { break }
      i = i + 1
    }
  }

  { role, content, reasoning_content, tool_calls }
}

/// Parse a single tool call from JSON
fn parse_single_tool_call(tc_json : String) -> ToolCall {
  let id = parse_json_string_field(tc_json, "id", "")
  let type_ = parse_json_string_field(tc_json, "type", "")

  // Extract function object
  let func_pattern = "\"function\":{"
  let func_idx = tc_json.find(func_pattern)
  let function = if func_idx >= 0 {
    let func_rest = tc_json.substring(func_idx + func_pattern.length() - 1, tc_json.length())
    let name = parse_json_string_field(func_rest, "name", "")
    let arguments = parse_json_string_field(func_rest, "arguments", "")
    { name, arguments }
  } else {
    { name: "", arguments: "" }
  }

  { id, type_, function }
}

/// Helper: convert float to string (complete implementation)
fn float_to_string(f : Float) -> String {
  // Handle common float values with exact representation
  if f == 0.0 { "0.0" }
  else if f == 1.0 { "1.0" }
  else if f == -1.0 { "-1.0" }
  else if f == 0.5 { "0.5" }
  else if f == 0.25 { "0.25" }
  else if f == 0.75 { "0.75" }
  else if f == 1.5 { "1.5" }
  else if f == 2.0 { "2.0" }
  else if f == 0.7 { "0.7" }
  else if f == 0.3 { "0.3" }
  else if f == 0.1 { "0.1" }
  else if f == 0.9 { "0.9" }
  else if f == 0.8 { "0.8" }
  else if f == 0.6 { "0.6" }
  else if f == 0.4 { "0.4" }
  else if f == 0.2 { "0.2" }
  else if f == 0.0 { "0.0" }
  else {
    // General conversion: handle any float
    let is_negative = f < 0.0
    let abs_f = if is_negative { -.f } else { f }

    // Extract integer part
    let int_part = float_to_int(abs_f)

    // Extract fractional part (up to 6 decimal places)
    let frac_part = float_to_int((abs_f -. (int_part as Float)) *. 1000000.0)

    // Build string
    let int_str = if is_negative { "-" ++ int_to_string(int_part) } else { int_to_string(int_part) }

    if frac_part == 0 {
      int_str ++ ".0"
    } else {
      // Remove trailing zeros from fractional part
      let mut frac_str = int_to_string(frac_part)
      while frac_str != "" && frac_str[frac_str.length() - 1] == '0' {
        frac_str = frac_str.substring(0, frac_str.length() - 1)
      }
      int_str ++ "." ++ (if frac_str == "" { "0" } else { frac_str })
    }
  }
}

/// Convert float to int (truncates)
fn float_to_int(f : Float) -> Int {
  // Simple truncation for positive floats
  // Note: MoonBit Float doesn't have direct int conversion, use approximation
  if f < 0.0 {
    0 // Handle negative case
  } else {
    let mut result = 0
    while (result as Float) <= f {
      result = result + 1
    }
    result - 1
  }
}

/// Helper: convert bool to string
fn bool_to_string(b : Bool) -> String {
  if b { "true" } else { "false" }
}

/// Helper: convert int to string (complete implementation)
fn int_to_string(i : Int) -> String {
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
  else if i == 2048 { "2048" }
  else {
    // General conversion for arbitrary integers
    if i < 0 {
      "-" ++ uint_to_string(-i)
    } else {
      uint_to_string(i)
    }
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

/// Helper: convert int64 to string
fn int64_to_string(i : Int64) -> String {
  // Int64 to string conversion
  let value = if i < 0L { -i } else { i }
  let str_val = uint64_to_string(value)
  if i < 0L { "-" ++ str_val } else { str_val }
}

/// Convert unsigned int64 to string
fn uint64_to_string(n : Int64) -> String {
  if n == 0L { "0" } else {
    let mut digits = []
    let mut num = n
    while num > 0L {
      let digit = num % 10L
      digits = [digit64_to_char(digit)] ++ digits
      num = num / 10L
    }
    let mut result = ""
    for d in digits {
      result = result ++ d
    }
    result
  }
}

/// Convert digit to char for int64
fn digit64_to_char(d : Int64) -> String {
  match d {
    0L => "0"
    1L => "1"
    2L => "2"
    3L => "3"
    4L => "4"
    5L => "5"
    6L => "6"
    7L => "7"
    8L => "8"
    _ => "9"
  }
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
  // Generate a random-ish ID using timestamp and counter
  let timestamp = current_timestamp()
  let ts_str = int64_to_string(timestamp)
  // Simple hash for uniqueness
  let hash = string_hash(ts_str ++ "chat")
  "chatcmpl-" ++ ts_str ++ "-" ++ int_to_string(hash)
}

/// Simple string hash function
fn string_hash(s : String) -> Int {
  let mut hash = 5381
  let bytes = s.to_utf8_bytes()
  for b in bytes {
    hash = ((hash << 5) + hash) + (b as Int)
  }
  hash
}

/// Get current Unix timestamp as Int64
pub fn current_timestamp() -> Int64 {
  // Returns current Unix timestamp in seconds as Int64
  // Note: In production, this would call the system time via FFI
  // For now, use a simple implementation based on environment or default
  0L // Placeholder - would need WASI time API in production
}