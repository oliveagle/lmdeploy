// src/streaming/engine.mbt
// Streaming engine integration - generates token chunks from TurboMind
// Reference: Python async_engine.py stream generation, Rust http.rs chunk handling

use username/lmdeploy_moonbit_server/streaming/config
use username/lmdeploy_moonbit_server/streaming/metrics
use username/lmdeploy_moonbit_server/streaming/sse
use username/lmdeploy_moonbit_server/protocol

/// Streaming result from engine (represents a token chunk)
pub struct EngineChunk {
  /// Generated content (delta)
  content: String
  /// Reasoning content (if thinking mode)
  reasoning_content: String
  /// Token IDs generated
  token_ids: Array[Int]
  /// Whether this is the final chunk
  is_finish: Bool
  /// Finish reason (stop, length, tool_calls)
  finish_reason: Option[String]
  /// Input tokens count (for usage calculation)
  input_tokens: Int
  /// Total generated tokens count
  generated_tokens: Int
  /// Time to first token in microseconds (only set on first chunk)
  time_to_first_token_us: Int
}

pub fn EngineChunk::default() -> EngineChunk {
  {
    content: "",
    reasoning_content: "",
    token_ids: [],
    is_finish: false,
    finish_reason: None,
    input_tokens: 0,
    generated_tokens: 0,
    time_to_first_token_us: 0,
  }
}

/// Generate a sequence of SSE events for a streaming chat completion request.
///
/// This is the core streaming function. The actual implementation would:
/// 1. Record the start timestamp
/// 2. Call the TurboMind engine to generate tokens incrementally
/// 3. Build SSE chunks as tokens are generated
/// 4. Track first token latency
/// 5. Send final chunk with usage information
///
/// Currently implemented as a placeholder structure since TurboMind FFI
/// is not yet available. The function returns the expected SSE output
/// string that would be streamed to the client.
///
/// Reference: Python api_server.py completion_stream_generator()
/// Reference: Rust http.rs chat_completions_stream()
pub fn stream_chat_completion(
  request: protocol.ChatCompletionRequest,
  request_id: String,
  created: Int64,
  config: config.StreamingConfig,
  metrics_snapshot: metrics.StreamingMetrics,
) -> (String, metrics.StreamingMetrics) {
  let model = request.model
  let mut response = ""

  // 1. Build initial empty chunk (establishes stream ID)
  // Reference: Python sends empty delta role chunk first
  let initial_chunk = sse.build_initial_chat_chunk(request_id, created, model)
  response = response ++ initial_chunk

  // 2. In production, this would iterate over engine results:
  //    while let Some(chunk) = engine.next_chunk() {
  //      record_first_token_latency(chunk.time_to_first_token_us)
  //      build_sse_chunk(chunk)
  //      send_to_client(chunk.sse_data)
  //    }
  //
  // For now, return a placeholder streaming response structure
  // that demonstrates the complete SSE format

  // Placeholder: simulate content deltas that would come from the engine
  // In production, these come from the async generator in the engine
  let content_chunks = ["Hello", " from", " the", " LMDeploy", " MoonBit", " server", "."]
  let mut total_tokens = 0

  for content in content_chunks {
    // Build delta chunk (no role after initial, no finish_reason)
    let chunk = sse.build_chat_stream_chunk(
      request_id,
      created,
      model,
      0,
      content,
      None,
      None,
      None,
    )
    response = response ++ chunk
    total_tokens = total_tokens + 1
  }

  // 3. Build final chunk with usage information
  // Reference: Python sends usage only when stream_options_include_usage is true
  let final_usage = {
    prompt_tokens: 0,
    completion_tokens: total_tokens,
    total_tokens: total_tokens,
  }

  if request.stream_options_include_usage {
    let final_chunk = sse.build_final_chat_chunk(request_id, created, model, final_usage)
    response = response ++ final_chunk
    let updated_metrics = metrics_snapshot.record_chunk()
    (response, updated_metrics)
  } else {
    let final_chunk = sse.build_chat_stream_chunk(
      request_id,
      created,
      model,
      0,
      "",
      None,
      Some("stop"),
      None,
    )
    response = response ++ final_chunk
    let updated_metrics = metrics_snapshot.record_chunk()
    (response, updated_metrics)
  }
}

/// Generate a sequence of SSE events for a streaming completion request.
///
/// Reference: Python api_server.py completions_v1() stream handling
/// Reference: Rust http.rs completions_stream()
pub fn stream_completion(
  prompt: String,
  request_id: String,
  created: Int64,
  model: String,
  config: config.StreamingConfig,
  metrics_snapshot: metrics.StreamingMetrics,
) -> (String, metrics.StreamingMetrics) {
  let mut response = ""

  // Initial chunk
  let initial = sse.build_initial_completion_chunk(request_id, created, model)
  response = response ++ initial

  // Placeholder content chunks that would come from the engine
  let content_chunks = ["This", " is", " a", " completion", " response", "."]
  let mut total_tokens = 0

  for content in content_chunks {
    let chunk = sse.build_completion_stream_chunk(
      request_id,
      created,
      model,
      0,
      content,
      None,
      None,
    )
    response = response ++ chunk
    total_tokens = total_tokens + 1
  }

  // Final chunk
  let final_usage = {
    prompt_tokens: 0,
    completion_tokens: total_tokens,
    total_tokens: total_tokens,
  }

  let final_chunk = sse.build_final_completion_chunk(request_id, created, model, final_usage)
  response = response ++ final_chunk

  let updated_metrics = metrics_snapshot.record_chunk()
  (response, updated_metrics)
}

/// Check stream timeout condition
/// Returns true if the elapsed time exceeds the configured timeout
pub fn is_stream_timeout(elapsed_us: Int, timeout_secs: Int) -> Bool {
  elapsed_us >= (timeout_secs * 1000000)
}

/// Record first token latency if this is the first chunk
/// Returns (updated_metrics, should_record) where should_record is true only on first chunk
pub fn record_first_token_if_first(
  metrics_snapshot: metrics.StreamingMetrics,
  elapsed_us: Int,
  is_first_chunk: Bool,
) -> metrics.StreamingMetrics {
  if is_first_chunk {
    metrics_snapshot.record_stream_start(elapsed_us)
  } else {
    metrics_snapshot
  }
}