//! Metrics module for LMDeploy Rust Server
//!
//! Provides Prometheus-compatible metrics for monitoring request latency,
//! throughput, cache hit rates, and streaming performance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Streaming metrics for tracking first token latency and chunk delivery
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub total_streams: AtomicU64,
    pub first_token_latency_ms: AtomicU64,
    pub total_chunks_sent: AtomicU64,
    pub stream_timeouts: AtomicU64,
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self {
            total_streams: AtomicU64::new(0),
            first_token_latency_ms: AtomicU64::new(0),
            total_chunks_sent: AtomicU64::new(0),
            stream_timeouts: AtomicU64::new(0),
        }
    }

    /// Record a stream start with first token latency
    pub fn record_stream_start(&self, latency_ms: u64) {
        self.total_streams.fetch_add(1, Ordering::Relaxed);
        self.first_token_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
    }

    /// Record a chunk sent
    pub fn record_chunk(&self) {
        self.total_chunks_sent.fetch_add(1, Ordering::Relaxed);
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
        }
    }
}

impl Default for StreamMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of streaming metrics at a point in time
#[derive(Debug, Clone)]
pub struct StreamMetricsSnapshot {
    pub total_streams: u64,
    pub avg_first_token_latency_ms: f64,
    pub total_chunks_sent: u64,
    pub stream_timeouts: u64,
}

/// Global metrics container
#[derive(Debug, Clone)]
pub struct AppMetrics {
    pub streams: StreamMetrics,
}

impl AppMetrics {
    pub fn new() -> Self {
        Self {
            streams: StreamMetrics::new(),
        }
    }
}

impl Default for AppMetrics {
    fn default() -> Self {
        Self::new()
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
}