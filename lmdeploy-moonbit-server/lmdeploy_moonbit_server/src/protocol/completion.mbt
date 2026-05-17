// src/protocol/completion.mbt
// Text completion protocol definitions (OpenAI-compatible)
// Reference: lmdeploy/serve/openai/protocol.py

use protocol.json_util

/// Text completion request (legacy, non-chat)
pub struct CompletionRequest {
  model: String
  prompt: String
  suffix: Option[String]
  max_tokens: Int
  temperature: Float
  top_p: Float
  top_k: Int
  n: Int
  stop: Array[String]
  stream: Bool
  logprobs: Int
  echo: Bool
  seed: Int
  presence_penalty: Float
  frequency_penalty: Float
  repetition_penalty: Float
  ignore_eos: Bool
  skip_special_tokens: Bool
  spaces_between_special_tokens: Bool
}

pub fn CompletionRequest::default(model: String, prompt: String) -> CompletionRequest {
  {
    model,
    prompt,
    suffix: None,
    max_tokens: 2048,
    temperature: 0.7,
    top_p: 1.0,
    top_k: 40,
    n: 1,
    stop: [],
    stream: false,
    logprobs: 0,
    echo: false,
    seed: 0,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    repetition_penalty: 1.0,
    ignore_eos: false,
    skip_special_tokens: true,
    spaces_between_special_tokens: true,
  }
}

/// Text completion response
pub struct CompletionResponse {
  id: String
  object: String
  created: Int64
  model: String
  choices: Array[CompletionResponseChoice]
  usage: UsageInfo
}

/// Single choice in completion response
pub struct CompletionResponseChoice {
  index: Int
  text: String
  logprobs: Option[CompletionTokenLogprobs]
  finish_reason: Option[String]
}

/// Token logprobs for completion
pub struct CompletionTokenLogprobs {
  tokens: Array[String]
  token_logprobs: Array[Float]
  top_logprobs: Array[Map[String, Float]]
  text_offset: Array[Int]
}

/// Text completion stream response
pub struct CompletionStreamResponse {
  id: String
  object: String
  created: Int64
  model: String
  choices: Array[CompletionStreamChoice]
  usage: Option[UsageInfo]
}

/// Stream choice in completion response
pub struct CompletionStreamChoice {
  index: Int
  text: String
  logprobs: Option[CompletionTokenLogprobs]
  finish_reason: Option[String]
}

/// Batch completion request
pub struct BatchCompletionRequest {
  requests: Array[CompletionRequest]
  batch_size: Int
}

/// Batch completion response
pub struct BatchCompletionResponse {
  responses: Array[CompletionResponse]
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

/// Serialize CompletionRequest to JSON string
pub fn CompletionRequest::to_json(self : CompletionRequest) -> String {
  let mut stop_json = "["
  let mut first_stop = true
  for s in self.stop {
    if !first_stop { stop_json = stop_json ++ ", " }
    stop_json = stop_json ++ "\"" ++ s ++ "\""
    first_stop = false
  }
  stop_json = stop_json ++ "]"

  "{
  \"model\":\"" ++ self.model ++ "\",
  \"prompt\":\"" ++ self.prompt ++ "\",
  \"max_tokens\":" ++ int_to_string(self.max_tokens) ++ ",
  \"temperature\":" ++ float_to_string(self.temperature) ++ ",
  \"top_p\":" ++ float_to_string(self.top_p) ++ ",
  \"top_k\":" ++ int_to_string(self.top_k) ++ ",
  \"n\":" ++ int_to_string(self.n) ++ ",
  \"stop\":" ++ stop_json ++ ",
  \"stream\":" ++ bool_to_string(self.stream) ++ ",
  \"logprobs\":" ++ int_to_string(self.logprobs) ++ ",
  \"echo\":" ++ bool_to_string(self.echo) ++ ",
  \"presence_penalty\":" ++ float_to_string(self.presence_penalty) ++ ",
  \"frequency_penalty\":" ++ float_to_string(self.frequency_penalty) ++ ",
  \"repetition_penalty\":" ++ float_to_string(self.repetition_penalty) ++ "
}"
}

/// Serialize CompletionResponse to JSON string
pub fn CompletionResponse::to_json(self : CompletionResponse) -> String {
  let mut choices_json = "["
  let mut first_choice = true
  for choice in self.choices {
    if !first_choice { choices_json = choices_json ++ ", " }
    choices_json = choices_json ++ "{\"index\":" ++ int_to_string(choice.index) ++ ",\"text\":\"" ++ choice.text ++ "\""
    let finish = match choice.finish_reason {
      Some(r) => r
      None => "stop"
    }
    choices_json = choices_json ++ ",\"finish_reason\":\"" ++ finish ++ "\"}"
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

/// Parse CompletionRequest from JSON string
pub fn CompletionRequest::from_json(json : String) -> CompletionRequest {
  {
    model: parse_json_string_field(json, "model", ""),
    prompt: parse_json_string_field(json, "prompt", ""),
    suffix: match parse_json_string_field(json, "suffix", "") {
      "" => None
      s => Some(s)
    },
    max_tokens: parse_json_int(json, "max_tokens", 2048),
    temperature: parse_json_float(json, "temperature", 0.7),
    top_p: parse_json_float(json, "top_p", 1.0),
    top_k: parse_json_int(json, "top_k", 40),
    n: parse_json_int(json, "n", 1),
    stop: parse_json_string_array(json, "stop"),
    stream: parse_json_bool(json, "stream", false),
    logprobs: parse_json_int(json, "logprobs", 0),
    echo: parse_json_bool(json, "echo", false),
    seed: parse_json_int(json, "seed", 0),
    presence_penalty: parse_json_float(json, "presence_penalty", 0.0),
    frequency_penalty: parse_json_float(json, "frequency_penalty", 0.0),
    repetition_penalty: parse_json_float(json, "repetition_penalty", 1.0),
    ignore_eos: parse_json_bool(json, "ignore_eos", false),
    skip_special_tokens: parse_json_bool(json, "skip_special_tokens", true),
    spaces_between_special_tokens: parse_json_bool(json, "spaces_between_special_tokens", true),
  }
}

/// Helper: convert int to string (complete implementation)
fn int_to_string(i : Int) -> String {
  if i == 0 { "0" }
  else if i == 1 { "1" }
  else if i == 40 { "40" }
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
  if i < 0L { "-" ++ uint64_to_string(-i) } else { uint64_to_string(i) }
}

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

/// Helper: convert float to string (complete implementation)
fn float_to_string(f : Float) -> String {
  if f == 0.0 { "0.0" }
  else if f == 1.0 { "1.0" }
  else if f == -1.0 { "-1.0" }
  else if f == 0.5 { "0.5" }
  else if f == 0.7 { "0.7" }
  else {
    let is_negative = f < 0.0
    let abs_f = if is_negative { -.f } else { f }
    let int_part = float_to_int(abs_f)
    let frac_part = float_to_int((abs_f -. (int_part as Float)) *. 1000000.0)

    let int_str = if is_negative { "-" ++ int_to_string(int_part) } else { int_to_string(int_part) }

    if frac_part == 0 {
      int_str ++ ".0"
    } else {
      let mut frac_str = int_to_string(frac_part)
      while frac_str != "" && frac_str[frac_str.length() - 1] == '0' {
        frac_str = frac_str.substring(0, frac_str.length() - 1)
      }
      int_str ++ "." ++ (if frac_str == "" { "0" } else { frac_str })
    }
  }
}

fn float_to_int(f : Float) -> Int {
  if f < 0.0 { 0 } else {
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

/// Generate a unique request ID
pub fn create_request_id() -> String {
  let timestamp = current_timestamp()
  let ts_str = int64_to_string(timestamp)
  let hash = string_hash(ts_str ++ "comp")
  "cmpl-" ++ ts_str ++ "-" ++ int_to_string(hash)
}

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
  0L // Would need WASI time API in production
}
