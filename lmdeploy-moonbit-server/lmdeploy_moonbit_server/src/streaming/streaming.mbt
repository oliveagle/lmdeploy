// src/streaming/streaming.mbt
// Streaming module - main entry point for streaming functionality
// Aggregates config, metrics, SSE, and engine streaming

mod config
mod metrics
mod sse
mod engine

pub fn StreamingConfig::default() -> config.StreamingConfig {
  config.StreamingConfig::default()
}

pub fn StreamingConfig::from_env() -> config.StreamingConfig {
  config.StreamingConfig::from_env()
}

pub fn StreamingMetrics::new() -> metrics.StreamingMetrics {
  metrics.StreamingMetrics::new()
}

pub fn StreamingMetrics::to_prometheus(self : metrics.StreamingMetrics) -> String {
  metrics.StreamingMetrics::to_prometheus(self)
}

pub fn StreamingMetrics::to_json(self : metrics.StreamingMetrics) -> String {
  metrics.StreamingMetrics::to_json(self)
}

pub fn stream_chat_completion(
  request: protocol.ChatCompletionRequest,
  request_id: String,
  created: Int64,
  stream_config: config.StreamingConfig,
  stream_metrics: metrics.StreamingMetrics,
) -> (String, metrics.StreamingMetrics) {
  engine.stream_chat_completion(request, request_id, created, stream_config, stream_metrics)
}

pub fn stream_completion(
  prompt: String,
  request_id: String,
  created: Int64,
  model: String,
  stream_config: config.StreamingConfig,
  stream_metrics: metrics.StreamingMetrics,
) -> (String, metrics.StreamingMetrics) {
  engine.stream_completion(prompt, request_id, created, model, stream_config, stream_metrics)
}

pub fn build_initial_chat_chunk(
  id: String,
  created: Int64,
  model: String,
) -> String {
  sse.build_initial_chat_chunk(id, created, model)
}

pub fn build_initial_completion_chunk(
  id: String,
  created: Int64,
  model: String,
) -> String {
  sse.build_initial_completion_chunk(id, created, model)
}

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
  sse.build_chat_stream_chunk(id, created, model, index, delta_content, delta_role, finish_reason, usage)
}

pub fn build_completion_stream_chunk(
  id: String,
  created: Int64,
  model: String,
  index: Int,
  delta_text: String,
  finish_reason: Option[String],
  usage: Option[protocol.UsageInfo],
) -> String {
  sse.build_completion_stream_chunk(id, created, model, index, delta_text, finish_reason, usage)
}

pub fn build_final_chat_chunk(
  id: String,
  created: Int64,
  model: String,
  usage: protocol.UsageInfo,
) -> String {
  sse.build_final_chat_chunk(id, created, model, usage)
}

pub fn build_final_completion_chunk(
  id: String,
  created: Int64,
  model: String,
  usage: protocol.UsageInfo,
) -> String {
  sse.build_final_completion_chunk(id, created, model, usage)
}

pub fn sse_data_event(json_data: String) -> String {
  sse.sse_data_event(json_data)
}

pub fn sse_done_event() -> String {
  sse.sse_done_event()
}

pub fn sse_keepalive_comment(comment: String) -> String {
  sse.sse_keepalive_comment(comment)
}

pub fn is_stream_timeout(elapsed_us: Int, timeout_secs: Int) -> Bool {
  engine.is_stream_timeout(elapsed_us, timeout_secs)
}
