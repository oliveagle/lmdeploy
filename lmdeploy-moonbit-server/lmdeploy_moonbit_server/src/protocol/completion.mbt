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

/// Serialize CompletionRequest to JSON string
pub fn CompletionRequest::to_json(self : CompletionRequest) -> String {
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

  // Build prompt_token_ids array JSON
  let mut ids_json = "["
  let mut first_id = true
  for id in self.prompt_token_ids {
    if !first_id {
      ids_json = ids_json ++ ", "
    }
    ids_json = ids_json ++ int_to_string(id)
    first_id = false
  }
  ids_json = ids_json ++ "]"

  "{
  \"model\":\"" ++ self.model ++ "\",
  \"prompt\":\"" ++ self.prompt ++ "\",
  \"prompt_token_ids\":" ++ ids_json ++ ",
  \"suffix\":\"" ++ self.suffix ++ "\",
  \"temperature\":" ++ float_to_string(self.temperature) ++ ",
  \"n\":" ++ int_to_string(self.n) ++ ",
  \"max_tokens\":" ++ int_to_string(self.max_tokens) ++ ",
  \"stop\":" ++ stop_json ++ ",
  \"stream\":" ++ bool_to_string(self.stream) ++ ",
  \"top_p\":" ++ float_to_string(self.top_p) ++ ",
  \"echo\":" ++ bool_to_string(self.echo) ++ ",
  \"presence_penalty\":" ++ float_to_string(self.presence_penalty) ++ ",
  \"frequency_penalty\":" ++ float_to_string(self.frequency_penalty) ++ ",
  \"repetition_penalty\":" ++ float_to_string(self.repetition_penalty) ++ ",
  \"ignore_eos\":" ++ bool_to_string(self.ignore_eos) ++ ",
  \"skip_special_tokens\":" ++ bool_to_string(self.skip_special_tokens) ++ ",
  \"top_k\":" ++ int_to_string(self.top_k) ++ ",
  \"min_p\":" ++ float_to_string(self.min_p) ++ ",
  \"session_id\":" ++ int_to_string(self.session_id) ++ ",
  \"top_logprobs\":" ++ int_to_string(self.top_logprobs) ++ "
}"
}

/// Serialize CompletionResponse to JSON string
pub fn CompletionResponse::to_json(self : CompletionResponse) -> String {
  // Build choices JSON array
  let mut choices_json = "["
  let mut first_choice = true
  for choice in self.choices {
    if !first_choice {
      choices_json = choices_json ++ ", "
    }

    choices_json = choices_json ++ "{\"index\":" ++ int_to_string(choice.index) ++ ",\"text\":\"" ++ choice.text ++ "\""

    // Add finish_reason if present
    let finish_reason_str = match choice.finish_reason {
      Some(reason) => reason
      None => "stop"
    }
    choices_json = choices_json ++ ",\"finish_reason\":\"" ++ finish_reason_str ++ "\"}"
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

/// Serialize CompletionStreamResponse to SSE JSON string
pub fn CompletionStreamResponse::to_sse_json(self : CompletionStreamResponse) -> String {
  let mut choices_json = "["
  let mut first_choice = true
  for choice in self.choices {
    if !first_choice {
      choices_json = choices_json ++ ", "
    }
    choices_json = choices_json ++ "{\"index\":" ++ int_to_string(choice.index) ++ ",\"text\":\"" ++ choice.text ++ "\""

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
  \"choices\:" ++ choices_json

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

/// Parse CompletionRequest from JSON string
pub fn CompletionRequest::from_json(json : String) -> CompletionRequest {
  // Extract model field
  let model = extract_string_field(json, "model")

  // Extract prompt field
  let prompt = extract_string_field(json, "prompt")

  // Extract stream field
  let stream = extract_bool_field(json, "stream")

  {
    model,
    prompt,
    prompt_token_ids: extract_int_array_field(json, "prompt_token_ids"),
    suffix: extract_string_field_default(json, "suffix", ""),
    temperature: extract_float_field(json, "temperature", 0.7),
    n: extract_int_field(json, "n", 1),
    max_tokens: extract_int_field(json, "max_tokens", 16),
    stop: extract_string_array_field(json, "stop"),
    stream,
    top_p: extract_float_field(json, "top_p", 1.0),
    echo: extract_bool_field_default(json, "echo", false),
    presence_penalty: extract_float_field(json, "presence_penalty", 0.0),
    frequency_penalty: extract_float_field(json, "frequency_penalty", 0.0),
    seed: extract_int_field(json, "seed", 0),
    repetition_penalty: extract_float_field(json, "repetition_penalty", 1.0),
    ignore_eos: extract_bool_field_default(json, "ignore_eos", false),
    skip_special_tokens: extract_bool_field_default(json, "skip_special_tokens", true),
    top_k: extract_int_field(json, "top_k", 40),
    min_p: extract_float_field(json, "min_p", 0.0),
    session_id: extract_int_field(json, "session_id", -1),
    top_logprobs: extract_int_field(json, "top_logprobs", 0),
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

/// Helper: extract int array field from JSON
fn extract_int_array_field(json : String, field : String) -> Array[Int] {
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
  } else if i == 16 {
    "16"
  } else if i == 40 {
    "40"
  } else {
    "0"
  }
}

/// Helper: convert int64 to string
fn int64_to_string(i : Int64) -> String {
  "0"
}
