//! Metrics module for LMDeploy Rust Server
//!
//! Provides Prometheus-compatible metrics for monitoring request latency,
//! throughput, cache hit rates, and streaming performance.
//!
//! Streaming metrics collection mirrors Python's RequestMetrics / EngineEvent:
//! - `EventType` enum: QUEUED, SCHEDULED, PREEMPTED (matches Python messages.py)
//! - `EngineEvent`: timestamped engine lifecycle event
//! - `StreamRequestMetrics`: per-request metrics with token_timestamp + engine_events

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use std::net::SocketAddr;

/// Engine event types matching Python `lmdeploy.messages.EventType`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineEventType {
    /// Request has been enqueued and is waiting in the queue
    Queued,
    /// Request has been scheduled for inference
    Scheduled,
    /// Request has been preempted from the engine
    Preempted,
}

impl EngineEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "QUEUED",
            Self::Scheduled => "SCHEDULED",
            Self::Preempted => "PREEMPTED",
        }
    }
}

/// A single engine lifecycle event, matching Python `lmdeploy.messages.EngineEvent`
#[derive(Debug, Clone, Copy)]
pub struct EngineEvent {
    pub event_type: EngineEventType,
    /// Wall-clock timestamp in seconds (same semantics as Python's `time.time()`)
    pub timestamp_secs: f64,
}

impl EngineEvent {
    pub fn new(event_type: EngineEventType, timestamp_secs: f64) -> Self {
        Self {
            event_type,
            timestamp_secs,
        }
    }

    pub fn now(event_type: EngineEventType) -> Self {
        Self {
            event_type,
            timestamp_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before epoch")
                .as_secs_f64(),
        }
    }
}

/// Per-request streaming metrics matching Python `lmdeploy.messages.RequestMetrics`
///
/// Tracks wall-clock token timestamps and engine lifecycle events for each
/// streaming request, enabling downstream calculation of TTFT, ITL, and TPOT.
#[derive(Debug, Clone)]
pub struct StreamRequestMetrics {
    /// Wall-clock time (seconds) of the most recent token generation
    pub token_timestamp_secs: f64,
    /// Engine lifecycle events collected during this request
    pub engine_events: Vec<EngineEvent>,
}

impl StreamRequestMetrics {
    pub fn new() -> Self {
        Self {
            token_timestamp_secs: Self::wall_time_secs(),
            engine_events: Vec::new(),
        }
    }

    /// Record an engine event on this request's timeline
    pub fn record_event(&mut self, event_type: EngineEventType) {
        self.engine_events.push(EngineEvent::now(event_type));
    }

    /// Update the token timestamp when a new token is generated
    pub fn mark_token_generated(&mut self) {
        self.token_timestamp_secs = Self::wall_time_secs();
    }

    /// Get the current wall-clock time as seconds since epoch
    fn wall_time_secs() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_secs_f64()
    }
}

impl Default for StreamRequestMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the Prometheus metrics exporter
pub fn init_metrics(config: &crate::config::MetricsConfig) {
    if !config.enabled {
        return;
    }

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("invalid metrics address");

    metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()
        .expect("failed to install Prometheus recorder");
}

/// Request duration histogram (seconds)
pub fn record_request_duration(duration_secs: f64) {
    metrics::histogram!("request_duration_seconds").record(duration_secs);
}

/// Prefill duration histogram (seconds)
pub fn record_prefill_duration(duration_secs: f64) {
    metrics::histogram!("prefill_duration_seconds").record(duration_secs);
}

/// Decode throughput (tokens per second)
pub fn record_decode_tokens_per_second(tokens_per_sec: f64) {
    metrics::histogram!("decode_tokens_per_second").record(tokens_per_sec);
}

/// Increment total requests counter
pub fn increment_requests_total() {
    metrics::counter!("requests_total").increment(1);
}

/// Increment total generated tokens counter
pub fn increment_tokens_generated_total(n_tokens: u64) {
    metrics::counter!("tokens_generated_total").increment(n_tokens);
}

/// Record a request with timing for full request lifecycle
pub struct RequestTimer {
    start: Instant,
}

impl RequestTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Record the request and return elapsed seconds
    pub fn finish(self) -> f64 {
        let secs = self.start.elapsed().as_secs_f64();
        record_request_duration(secs);
        increment_requests_total();
        secs
    }
}

impl Default for RequestTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming metrics for tracking first token latency and chunk delivery
#[derive(Debug)]
pub struct StreamMetrics {
    pub total_streams: AtomicU64,
    pub first_token_latency_ms: AtomicU64,
    pub total_chunks_sent: AtomicU64,
    pub stream_timeouts: AtomicU64,
    pub total_stream_tokens: AtomicU64,
}

impl Clone for StreamMetrics {
    fn clone(&self) -> Self {
        Self {
            total_streams: AtomicU64::new(self.total_streams.load(Ordering::Relaxed)),
            first_token_latency_ms: AtomicU64::new(
                self.first_token_latency_ms.load(Ordering::Relaxed),
            ),
            total_chunks_sent: AtomicU64::new(self.total_chunks_sent.load(Ordering::Relaxed)),
            stream_timeouts: AtomicU64::new(self.stream_timeouts.load(Ordering::Relaxed)),
            total_stream_tokens: AtomicU64::new(self.total_stream_tokens.load(Ordering::Relaxed)),
        }
    }
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self {
            total_streams: AtomicU64::new(0),
            first_token_latency_ms: AtomicU64::new(0),
            total_chunks_sent: AtomicU64::new(0),
            stream_timeouts: AtomicU64::new(0),
            total_stream_tokens: AtomicU64::new(0),
        }
    }

    /// Record a stream start with first token latency
    pub fn record_stream_start(&self, latency_ms: u64) {
        self.total_streams.fetch_add(1, Ordering::Relaxed);
        self.first_token_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
        // Record to Prometheus histogram as well
        metrics::histogram!("first_token_latency_seconds").record(latency_ms as f64 / 1000.0);
    }

    /// Record a chunk sent
    pub fn record_chunk(&self) {
        self.total_chunks_sent.fetch_add(1, Ordering::Relaxed);
        self.total_stream_tokens.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a stream timeout
    pub fn record_timeout(&self) {
        self.stream_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// Get aggregated metrics
    pub fn snapshot(&self) -> StreamMetricsSnapshot {
        let total_streams = self.total_streams.load(Ordering::Relaxed);
        let total_first_token_ms = self.first_token_latency_ms.load(Ordering::Relaxed);

        StreamMetricsSnapshot {
            total_streams,
            avg_first_token_latency_ms: if total_streams > 0 {
                total_first_token_ms as f64 / total_streams as f64
            } else {
                0.0
            },
            total_chunks_sent: self.total_chunks_sent.load(Ordering::Relaxed),
            stream_timeouts: self.stream_timeouts.load(Ordering::Relaxed),
            total_stream_tokens: self.total_stream_tokens.load(Ordering::Relaxed),
        }
    }
}

impl Default for StreamMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of streaming metrics at a point in time
#[derive(Debug, Clone, Serialize)]
pub struct StreamMetricsSnapshot {
    pub total_streams: u64,
    pub avg_first_token_latency_ms: f64,
    pub total_chunks_sent: u64,
    pub stream_timeouts: u64,
    pub total_stream_tokens: u64,
}

/// Global metrics container
#[derive(Debug, Clone)]
pub struct AppMetrics {
    pub streams: StreamMetrics,
    pub start_time: Instant,
}

impl AppMetrics {
    pub fn new() -> Self {
        Self {
            streams: StreamMetrics::new(),
            start_time: Instant::now(),
        }
    }

    /// Uptime in seconds
    pub fn uptime_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

impl Default for AppMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper guard for measuring prefill + decode timing
pub struct InferenceTimer {
    start: Instant,
    prompt_len: usize,
    label: &'static str,
}

impl InferenceTimer {
    pub fn new_prefill(prompt_len: usize) -> Self {
        Self {
            start: Instant::now(),
            prompt_len,
            label: "prefill",
        }
    }

    /// Record prefill duration
    pub fn finish_prefill(self) -> Duration {
        let elapsed = self.start.elapsed();
        record_prefill_duration(elapsed.as_secs_f64());
        tracing::info!(
            label = self.label,
            prompt_tokens = self.prompt_len,
            prefill_duration_ms = elapsed.as_millis(),
            "Inference prefill completed"
        );
        elapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_metrics() {
        let metrics = StreamMetrics::new();

        metrics.record_stream_start(15);
        metrics.record_stream_start(25);
        metrics.record_chunk();
        metrics.record_chunk();
        metrics.record_chunk();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_streams, 2);
        assert!((snapshot.avg_first_token_latency_ms - 20.0).abs() < 0.001);
        assert_eq!(snapshot.total_chunks_sent, 3);
    }

    #[tokio::test]
    async fn test_metrics_empty() {
        let metrics = StreamMetrics::new();
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_streams, 0);
        assert_eq!(snapshot.avg_first_token_latency_ms, 0.0);
    }

    #[tokio::test]
    async fn test_request_timer() {
        let timer = RequestTimer::new();
        tokio::time::sleep(Duration::from_millis(10)).await;
        let elapsed = timer.finish();
        assert!(elapsed > 0.009);
    }

    #[test]
    fn test_engine_event_type() {
        assert_eq!(EngineEventType::Queued.as_str(), "QUEUED");
        assert_eq!(EngineEventType::Scheduled.as_str(), "SCHEDULED");
        assert_eq!(EngineEventType::Preempted.as_str(), "PREEMPTED");
    }

    #[test]
    fn test_engine_event_new() {
        let event = EngineEvent::new(EngineEventType::Scheduled, 12345.0);
        assert_eq!(event.event_type, EngineEventType::Scheduled);
        assert_eq!(event.timestamp_secs, 12345.0);
    }

    #[test]
    fn test_engine_event_now() {
        let event = EngineEvent::now(EngineEventType::Queued);
        assert_eq!(event.event_type, EngineEventType::Queued);
        assert!(event.timestamp_secs > 0.0);
    }

    #[test]
    fn test_stream_request_metrics_new() {
        let metrics = StreamRequestMetrics::new();
        assert!(metrics.token_timestamp_secs > 0.0);
        assert_eq!(metrics.engine_events.len(), 0);
    }

    #[test]
    fn test_stream_request_metrics_record_event() {
        let mut metrics = StreamRequestMetrics::new();
        metrics.record_event(EngineEventType::Queued);
        metrics.record_event(EngineEventType::Scheduled);

        assert_eq!(metrics.engine_events.len(), 2);
        assert_eq!(metrics.engine_events[0].event_type, EngineEventType::Queued);
        assert_eq!(metrics.engine_events[1].event_type, EngineEventType::Scheduled);
    }

    #[test]
    fn test_stream_request_metrics_mark_token() {
        let mut metrics = StreamRequestMetrics::new();
        let first_ts = metrics.token_timestamp_secs;

        // Simulate small delay
        std::thread::sleep(Duration::from_millis(10));
        metrics.mark_token_generated();

        assert!(metrics.token_timestamp_secs > first_ts);
    }

    #[test]
    fn test_stream_request_metrics_default() {
        let metrics = StreamRequestMetrics::default();
        assert!(metrics.token_timestamp_secs > 0.0);
        assert_eq!(metrics.engine_events.len(), 0);
    }
}
