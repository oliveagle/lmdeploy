// src/streaming/sse.mbt
// SSE (Server-Sent Events) formatting utilities for streaming responses
// Reference: Python api_server.py completion_stream_generator(), Rust http.rs SSE handling

use username/lmdeploy_moonbit_server.protocol

/// SSE event wrapper for chat completion streaming
/// Format: "data: {...}\n\n"
pub struct SseEvent {
  event_type: String
  data: String
}

/// Create an SSE data event string
/// Format: "data: {...}\n\n"
pub fn sse_data_event(json_data: String) -> String {
  "data: " ++ json_data ++ "\n\n"
}

/// Create SSE [DONE] event to signal stream completion
/// Format: "data: [DONE]\n\n"
pub fn sse_done_event() -> String {
  "data: [DONE]\n\n"
}

/// Create SSE comment event (for keep-alive)
/// Format: ": <comment>\n\n"
pub fn sse_keepalive_comment(comment: String) -> String {
  ": " ++ comment ++ "\n\n"
}

/// Build chat completion stream chunk in SSE format
/// Reference: Python api_server.py -> create_stream_response_json()
pub fn build_chat_stream_chunk(
  id: String,
  created: Int64,
  model: String,
  index: Int,
  delta_content: String,
  delta_role: Option[String],
  finish_reason: Option[String],
  usage: Option[protocol.UsageInfo],
) -> String {
  // Build delta object
  let mut delta_json = "{"
  let mut first = true

  // Add role if present
  match delta_role {
    Some(role) => {
      if !first { delta_json = delta_json ++ "," }
      delta_json = delta_json ++ "\"role\":\"" ++ role ++ "\""
      first = false
    }
    None => ()
  }

  // Add content if present
  if delta_content.length() > 0 {
    if !first { delta_json = delta_json ++ "," }
    delta_json = delta_json ++ "\"content\":\"" ++ escape_json_string(delta_content) ++ "\""
    first = false
  }

  delta_json = delta_json ++ "}"

  // Build choice
  let mut choice_json = "\"index\":" ++ int_to_string(index) ++ ",\"delta\":" ++ delta_json

  // Add finish_reason if present
  match finish_reason {
    Some(reason) => {
      choice_json = choice_json ++ ",\"finish_reason\":\"" ++ reason ++ "\""
    }
    None => {
      choice_json = choice_json ++ ",\"finish_reason\":null"
    }
  }

  // Build choices array
  let choices_json = "[" ++ choice_json ++ "]"

  // Build base response
  let mut base_json = "\"id\":\"" ++ id ++ "\",
  \"object\":\"chat.completion.chunk\",
  \"created\":" ++ int64_to_string(created) ++ ",
  \"model\":\"" ++ model ++ "\",
  \"choices\":" ++ choices_json

  // Add usage if present (only in final chunk)
  match usage {
    Some(u) => {
      base_json = base_json ++ ",
  \"usage\":{\"prompt_tokens\":" ++ int_to_string(u.prompt_tokens) ++ ",\"completion_tokens\":" ++ int_to_string(u.completion_tokens) ++ ",\"total_tokens\":" ++ int_to_string(u.total_tokens) ++ "}"
    }
    None => {
      base_json = base_json ++ ",
  \"usage\":null"
    }
  }

  let json = "{" ++ base_json ++ "}"
  sse_data_event(json)
}

/// Build completion stream chunk in SSE format
/// Reference: Python api_server.py similar pattern for completion
pub fn build_completion_stream_chunk(
  id: String,
  created: Int64,
  model: String,
  index: Int,
  delta_text: String,
  finish_reason: Option[String],
  usage: Option[protocol.UsageInfo],
) -> String {
  // Build choice
  let mut choice_json = "\"index\":" ++ int_to_string(index) ++ ",\"text\":\"" ++ escape_json_string(delta_text) ++ "\""

  // Add finish_reason if present
  match finish_reason {
    Some(reason) => {
      choice_json = choice_json ++ ",\"finish_reason\":\"" ++ reason ++ "\""
    }
    None => {
      choice_json = choice_json ++ ",\"finish_reason\":null"
    }
  }

  // Build choices array
  let choices_json = "[" ++ choice_json ++ "]"

  // Build base response
  let mut base_json = "\"id\":\"" ++ id ++ "\",
  \"object\":\"text_completion\",
  \"created\":" ++ int64_to_string(created) ++ ",
  \"model\":\"" ++ model ++ "\",
  \"choices\":" ++ choices_json

  // Add usage if present (only in final chunk)
  match usage {
    Some(u) => {
      base_json = base_json ++ ",
  \"usage\":{\"prompt_tokens\":" ++ int_to_string(u.prompt_tokens) ++ ",\"completion_tokens\":" ++ int_to_string(u.completion_tokens) ++ ",\"total_tokens\":" ++ int_to_string(u.total_tokens) ++ "}"
    }
    None => {
      base_json = base_json ++ ",
  \"usage\":null"
    }
  }

  let json = "{" ++ base_json ++ "}"
  sse_data_event(json)
}

/// Build initial empty chunk for chat completion stream
/// This establishes the stream ID and model before deltas arrive
pub fn build_initial_chat_chunk(
  id: String,
  created: Int64,
  model: String,
) -> String {
  build_chat_stream_chunk(id, created, model, 0, "", Some("assistant"), None, None)
}

/// Build initial empty chunk for completion stream
pub fn build_initial_completion_chunk(
  id: String,
  created: Int64,
  model: String,
) -> String {
  build_completion_stream_chunk(id, created, model, 0, "", None, None)
}

/// Build final chunk with usage information
/// Usage is only sent in the final chunk per OpenAI API spec
pub fn build_final_chat_chunk(
  id: String,
  created: Int64,
  model: String,
  usage: protocol.UsageInfo,
) -> String {
  build_chat_stream_chunk(id, created, model, 0, "", None, Some("stop"), Some(usage))
}

/// Build final chunk for completion stream
pub fn build_final_completion_chunk(
  id: String,
  created: Int64,
  model: String,
  usage: protocol.UsageInfo,
) -> String {
  build_completion_stream_chunk(id, created, model, 0, "", Some("stop"), Some(usage))
}

/// Escape special characters in JSON string values
/// Handles: " -> \", \ -> \\, newline -> \n, carriage return -> \r, tab -> \t
fn escape_json_string(s: String) -> String {
  let mut result = ""
  let chars = s.to_bytes()
  for c in chars {
    match c {
      34U8 => result = result ++ "\\\"" // "
      92U8 => result = result ++ "\\\\" // \
      10U8 => result = result ++ "\\n"  // newline
      13U8 => result = result ++ "\\r"  // carriage return
      9U8 => result = result ++ "\\t"   // tab
      _ => result = result ++ String::from_byte(c)
    }
  }
  result
}

/// Helper: convert int to string
fn int_to_string(i: Int) -> String {
  if i == 0 { "0" } else if i == 1 { "1" } else { "0" }
}

/// Helper: convert int64 to string
fn int64_to_string(i: Int64) -> String {
  // TODO: Implement proper Int64 to string
  "0"
}