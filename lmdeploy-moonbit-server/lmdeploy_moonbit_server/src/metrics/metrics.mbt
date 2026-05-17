// src/metrics/metrics.mbt
// Prometheus metrics collection for LMDeploy MoonBit server
// Reference: lmdeploy/metrics/metrics_processor.py, lmdeploy-rust-server/src/metrics.rs
//
// Features:
// - Request counting (total, by endpoint, by status)
// - Request duration histograms
// - Token throughput metrics
// - Active requests gauge
// - Error tracking

/// Metric type enumeration
pub enum MetricType {
  Counter
  Gauge
  Histogram
  Summary
}

pub fn MetricType::to_string(self : MetricType) -> String {
  match self {
    Counter => "counter"
    Gauge => "gauge"
    Histogram => "histogram"
    Summary => "summary"
  }
}

/// HTTP method enumeration
pub enum HttpMethod {
  Get
  Post
  Put
  Delete
  Patch
  Options
  Head
}

pub fn HttpMethod::to_string(self : HttpMethod) -> String {
  match self {
    Get => "GET"
    Post => "POST"
    Put => "PUT"
    Delete => "DELETE"
    Patch => "PATCH"
    Options => "OPTIONS"
    Head => "HEAD"
  }
}

/// Histogram bucket configuration
pub struct HistogramBuckets {
  prefill_duration_ms: Array[Float]
  request_duration_ms: Array[Float]
  decode_tokens_per_sec: Array[Float]
}

pub fn HistogramBuckets::default() -> HistogramBuckets {
  {
    prefill_duration_ms: [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, +Inf],
    request_duration_ms: [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, +Inf],
    decode_tokens_per_sec: [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, +Inf],
  }
}

/// Request metrics tracked per request
pub struct RequestMetrics {
  endpoint: String
  method: HttpMethod
  status_code: Int
  duration_ms: Int
  prefill_duration_ms: Option[Int]
  decode_duration_ms: Option[Int]
  prompt_tokens: Int
  completion_tokens: Int
  cache_hit: Bool
  stream: Bool
}

pub fn RequestMetrics::new(endpoint: String, method: HttpMethod) -> RequestMetrics {
  {
    endpoint,
    method,
    status_code: 200,
    duration_ms: 0,
    prefill_duration_ms: None,
    decode_duration_ms: None,
    prompt_tokens: 0,
    completion_tokens: 0,
    cache_hit: false,
    stream: false,
  }
}

pub fn RequestMetrics::with_status(self: RequestMetrics, status_code: Int) -> RequestMetrics {
  { endpoint: self.endpoint, method: self.method, status_code, duration_ms: self.duration_ms, prefill_duration_ms: self.prefill_duration_ms, decode_duration_ms: self.decode_duration_ms, prompt_tokens: self.prompt_tokens, completion_tokens: self.completion_tokens, cache_hit: self.cache_hit, stream: self.stream }
}

pub fn RequestMetrics::with_duration(self: RequestMetrics, duration_ms: Int) -> RequestMetrics {
  { endpoint: self.endpoint, method: self.method, status_code: self.status_code, duration_ms, prefill_duration_ms: self.prefill_duration_ms, decode_duration_ms: self.decode_duration_ms, prompt_tokens: self.prompt_tokens, completion_tokens: self.completion_tokens, cache_hit: self.cache_hit, stream: self.stream }
}

pub fn RequestMetrics::with_prefill_duration(self: RequestMetrics, prefill_ms: Int) -> RequestMetrics {
  { endpoint: self.endpoint, method: self.method, status_code: self.status_code, duration_ms: self.duration_ms, prefill_duration_ms: Some(prefill_ms), decode_duration_ms: self.decode_duration_ms, prompt_tokens: self.prompt_tokens, completion_tokens: self.completion_tokens, cache_hit: self.cache_hit, stream: self.stream }
}

pub fn RequestMetrics::with_decode_duration(self: RequestMetrics, decode_ms: Int) -> RequestMetrics {
  { endpoint: self.endpoint, method: self.method, status_code: self.status_code, duration_ms: self.duration_ms, prefill_duration_ms: self.prefill_duration_ms, decode_duration_ms: Some(decode_ms), prompt_tokens: self.prompt_tokens, completion_tokens: self.completion_tokens, cache_hit: self.cache_hit, stream: self.stream }
}

pub fn RequestMetrics::with_tokens(self: RequestMetrics, prompt_tokens: Int, completion_tokens: Int) -> RequestMetrics {
  { endpoint: self.endpoint, method: self.method, status_code: self.status_code, duration_ms: self.duration_ms, prefill_duration_ms: self.prefill_duration_ms, decode_duration_ms: self.decode_duration_ms, prompt_tokens, completion_tokens, cache_hit: self.cache_hit, stream: self.stream }
}

pub fn RequestMetrics::with_cache_status(self: RequestMetrics, cache_hit: Bool) -> RequestMetrics {
  { endpoint: self.endpoint, method: self.method, status_code: self.status_code, duration_ms: self.duration_ms, prefill_duration_ms: self.prefill_duration_ms, decode_duration_ms: self.decode_duration_ms, prompt_tokens: self.prompt_tokens, completion_tokens: self.completion_tokens, cache_hit, stream: self.stream }
}

pub fn RequestMetrics::with_stream(self: RequestMetrics, stream: Bool) -> RequestMetrics {
  { endpoint: self.endpoint, method: self.method, status_code: self.status_code, duration_ms: self.duration_ms, prefill_duration_ms: self.prefill_duration_ms, decode_duration_ms: self.decode_duration_ms, prompt_tokens: self.prompt_tokens, completion_tokens: self.completion_tokens, cache_hit: self.cache_hit, stream }
}

/// Histogram bucket counts
pub struct Histogram {
  buckets: Array[Float]
  counts: Array[Int]
  sum: Float
  count: Int
}

pub fn Histogram::new(buckets: Array[Float]) -> Histogram {
  { buckets, counts: Array::make(buckets.length(), 0), sum: 0.0, count: 0 }
}

pub fn Histogram::observe(self: Histogram, value: Float) -> Histogram {
  let mut updated_counts = self.counts
  let mut idx = 0
  let bucket_count = self.buckets.length()

  while idx < bucket_count {
    if value <= self.buckets[idx] {
      let current = updated_counts[idx]
      updated_counts = updated_counts.update(idx, current + 1)
    }
    idx = idx + 1
  }

  { buckets: self.buckets, counts: updated_counts, sum: self.sum + value, count: self.count + 1 }
}

pub fn Histogram::to_prometheus(self: Histogram, name: String, labels: String) -> String {
  let mut result = "# HELP " ++ name ++ " " ++ name ++ "\n"
  result = result ++ "# TYPE " ++ name ++ " histogram\n"

  let mut idx = 0
  while idx < self.buckets.length() {
    let le = if self.buckets[idx] == +Inf { "+Inf" } else { float_to_string(self.buckets[idx]) }
    let label_suffix = if labels == "" { "" } else { "," ++ labels }
    result = result ++ name ++ "_bucket{le=\"" ++ le ++ "\"" ++ label_suffix ++ "} " ++ int_to_string(self.counts[idx]) ++ "\n"
    idx = idx + 1
  }

  let label_suffix = if labels == "" { "" } else { "," ++ labels }
  result = result ++ name ++ "_bucket{le=\"+Inf\"" ++ label_suffix ++ "} " ++ int_to_string(self.count) ++ "\n"
  result = result ++ name ++ "_sum" ++ label_suffix ++ " " ++ float_to_string(self.sum) ++ "\n"
  result = result ++ name ++ "_count" ++ label_suffix ++ " " ++ int_to_string(self.count) ++ "\n"

  result
}

/// Main metrics collector
pub struct MetricsCollector {
  /// Total HTTP requests by method, path, and status
  requests_total: Map[String, Int]
  /// Active requests gauge
  active_requests: Int
  /// Request duration histogram
  request_duration_seconds: Histogram
  /// Prefill duration histogram
  prefill_duration_seconds: Histogram
  /// Decode tokens per second histogram
  decode_tokens_per_second: Histogram
  /// Total tokens generated
  tokens_generated_total: Int
  /// Total prompt tokens processed
  prompt_tokens_total: Int
  /// Cache hits
  cache_hits_total: Int
  /// Cache misses
  cache_misses_total: Int
  /// Errors by type
  errors_total: Map[String, Int]
  /// Streaming requests
  streaming_requests_total: Int
  /// Non-streaming requests
  non_streaming_requests_total: Int
}

pub fn MetricsCollector::new() -> MetricsCollector {
  let buckets = HistogramBuckets::default()
  {
    requests_total: Map([], capacity=50),
    active_requests: 0,
    request_duration_seconds: Histogram::new(buckets.request_duration_ms.map(fn (x) { x / 1000.0 })),
    prefill_duration_seconds: Histogram::new(buckets.prefill_duration_ms.map(fn (x) { x / 1000.0 })),
    decode_tokens_per_second: Histogram::new(buckets.decode_tokens_per_sec),
    tokens_generated_total: 0,
    prompt_tokens_total: 0,
    cache_hits_total: 0,
    cache_misses_total: 0,
    errors_total: Map([], capacity=20),
    streaming_requests_total: 0,
    non_streaming_requests_total: 0,
  }
}

pub fn MetricsCollector::default() -> MetricsCollector {
  MetricsCollector::new()
}

/// Record a request start
pub fn MetricsCollector::start_request(self: MetricsCollector) -> MetricsCollector {
  { active_requests: self.active_requests + 1, requests_total: self.requests_total, request_duration_seconds: self.request_duration_seconds, prefill_duration_seconds: self.prefill_duration_seconds, decode_tokens_per_second: self.decode_tokens_per_second, tokens_generated_total: self.tokens_generated_total, prompt_tokens_total: self.prompt_tokens_total, cache_hits_total: self.cache_hits_total, cache_misses_total: self.cache_misses_total, errors_total: self.errors_total, streaming_requests_total: self.streaming_requests_total, non_streaming_requests_total: self.non_streaming_requests_total }
}

/// Record a completed request
pub fn MetricsCollector::record_request(self: MetricsCollector, metrics: RequestMetrics) -> MetricsCollector {
  let mut updated_self = self

  // Decrement active requests
  updated_self = { active_requests: updated_self.active_requests - 1, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }

  // Increment request counter for this endpoint/method/status
  let key = metrics.endpoint ++ "," ++ HttpMethod::to_string(metrics.method) ++ "," ++ int_to_string(metrics.status_code)
  let current = updated_self.requests_total.get(key).unwrap_or(0)
  updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total.insert(key, current + 1), request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }

  // Record duration
  updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds.observe(to_float(metrics.duration_ms) / 1000.0), prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }

  // Record prefill duration if available
  match metrics.prefill_duration_ms {
    Some(prefill_ms) => {
      updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds.observe(to_float(prefill_ms) / 1000.0), decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }
    }
    None => ()
  }

  // Record decode throughput if available
  match metrics.decode_duration_ms {
    Some(decode_ms) => {
      if decode_ms > 0 && metrics.completion_tokens > 0 {
        let tps = to_float(metrics.completion_tokens) / (to_float(decode_ms) / 1000.0)
        updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second.observe(tps), tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }
      }
    }
    None => ()
  }

  // Update token counters
  updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total + metrics.completion_tokens, prompt_tokens_total: updated_self.prompt_tokens_total + metrics.prompt_tokens, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }

  // Update cache stats
  if metrics.cache_hit {
    updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total + 1, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }
  } else {
    updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total + 1, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }
  }

  // Update streaming counters
  if metrics.stream {
    updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total + 1, non_streaming_requests_total: updated_self.non_streaming_requests_total }
  } else {
    updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total, streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total + 1 }
  }

  // Track errors
  if metrics.status_code >= 400 {
    let error_type = if metrics.status_code >= 500 { "5xx" } else { "4xx" }
    let current = updated_self.errors_total.get(error_type).unwrap_or(0)
    updated_self = { active_requests: updated_self.active_requests, requests_total: updated_self.requests_total, request_duration_seconds: updated_self.request_duration_seconds, prefill_duration_seconds: updated_self.prefill_duration_seconds, decode_tokens_per_second: updated_self.decode_tokens_per_second, tokens_generated_total: updated_self.tokens_generated_total, prompt_tokens_total: updated_self.prompt_tokens_total, cache_hits_total: updated_self.cache_hits_total, cache_misses_total: updated_self.cache_misses_total, errors_total: updated_self.errors_total.insert(error_type, current + 1), streaming_requests_total: updated_self.streaming_requests_total, non_streaming_requests_total: updated_self.non_streaming_requests_total }
  }

  updated_self
}

/// Record an error
pub fn MetricsCollector::record_error(self: MetricsCollector, error_type: String) -> MetricsCollector {
  let current = self.errors_total.get(error_type).unwrap_or(0)
  { active_requests: self.active_requests, requests_total: self.requests_total, request_duration_seconds: self.request_duration_seconds, prefill_duration_seconds: self.prefill_duration_seconds, decode_tokens_per_second: self.decode_tokens_per_second, tokens_generated_total: self.tokens_generated_total, prompt_tokens_total: self.prompt_tokens_total, cache_hits_total: self.cache_hits_total, cache_misses_total: self.cache_misses_total, errors_total: self.errors_total.insert(error_type, current + 1), streaming_requests_total: self.streaming_requests_total, non_streaming_requests_total: self.non_streaming_requests_total }
}

/// Export metrics in Prometheus format
pub fn MetricsCollector::to_prometheus(self: MetricsCollector) -> String {
  let mut result = ""

  // Active requests gauge
  result = result ++ "# HELP active_requests Number of currently active requests\n"
  result = result ++ "# TYPE active_requests gauge\n"
  result = result ++ "active_requests " ++ int_to_string(self.active_requests) ++ "\n\n"

  // Total requests counter
  result = result ++ "# HELP http_requests_total Total number of HTTP requests\n"
  result = result ++ "# TYPE http_requests_total counter\n"
  for (key, count) in self.requests_total {
    result = result ++ "http_requests_total{" ++ key ++ "} " ++ int_to_string(count) ++ "\n"
  }
  result = result ++ "\n"

  // Request duration histogram
  result = result ++ self.request_duration_seconds.to_prometheus("request_duration_seconds", "")

  // Prefill duration histogram
  result = result ++ "\n" ++ self.prefill_duration_seconds.to_prometheus("prefill_duration_seconds", "")

  // Decode tokens per second histogram
  result = result ++ "\n" ++ self.decode_tokens_per_second.to_prometheus("decode_tokens_per_second", "")

  // Token counters
  result = result ++ "\n# HELP tokens_generated_total Total number of tokens generated\n"
  result = result ++ "# TYPE tokens_generated_total counter\n"
  result = result ++ "tokens_generated_total " ++ int_to_string(self.tokens_generated_total) ++ "\n\n"

  result = result ++ "# HELP prompt_tokens_total Total number of prompt tokens processed\n"
  result = result ++ "# TYPE prompt_tokens_total counter\n"
  result = result ++ "prompt_tokens_total " ++ int_to_string(self.prompt_tokens_total) ++ "\n\n"

  // Cache stats
  result = result ++ "# HELP cache_hits_total Total number of cache hits\n"
  result = result ++ "# TYPE cache_hits_total counter\n"
  result = result ++ "cache_hits_total " ++ int_to_string(self.cache_hits_total) ++ "\n\n"

  result = result ++ "# HELP cache_misses_total Total number of cache misses\n"
  result = result ++ "# TYPE cache_misses_total counter\n"
  result = result ++ "cache_misses_total " ++ int_to_string(self.cache_misses_total) ++ "\n\n"

  // Streaming counters
  result = result ++ "# HELP streaming_requests_total Total number of streaming requests\n"
  result = result ++ "# TYPE streaming_requests_total counter\n"
  result = result ++ "streaming_requests_total " ++ int_to_string(self.streaming_requests_total) ++ "\n\n"

  result = result ++ "# HELP non_streaming_requests_total Total number of non-streaming requests\n"
  result = result ++ "# TYPE non_streaming_requests_total counter\n"
  result = result ++ "non_streaming_requests_total " ++ int_to_string(self.non_streaming_requests_total) ++ "\n\n"

  // Error counters
  result = result ++ "# HELP errors_total Total number of errors by type\n"
  result = result ++ "# TYPE errors_total counter\n"
  for (error_type, count) in self.errors_total {
    result = result ++ "errors_total{type=\"" ++ error_type ++ "\"} " ++ int_to_string(count) ++ "\n"
  }

  result
}

// Helper functions

fn int_to_string(i: Int) -> String {
  if i < 0 {
    "-" ++ int_to_string(-i)
  } else if i < 10 {
    match i {
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
  } else if i < 100 {
    let tens = i / 10
    let ones = i - (tens * 10)
    int_to_string(tens) ++ int_to_string(ones)
  } else if i < 1000 {
    let hundreds = i / 100
    let rest = i - (hundreds * 100)
    int_to_string(hundreds) ++ int_to_string(rest)
  } else if i < 10000 {
    let thousands = i / 1000
    let rest = i - (thousands * 1000)
    int_to_string(thousands) ++ int_to_string(rest)
  } else {
    let first = i / 1000
    int_to_string(first) ++ "000"
  }
}

fn float_to_string(f: Float) -> String {
  if f == +Inf { "+Inf" }
  else if f == 0.0 { "0.0" }
  else if f == 1.0 { "1.0" }
  else if f < 0.0 { "-" ++ float_to_string_positive(-.f) }
  else { float_to_string_positive(f) }
}

fn float_to_string_positive(f: Float) -> String {
  // Extract integer part
  let int_part = float_to_int(f)
  let frac_part = float_to_int((f -. (int_part as Float)) *. 1000000.0)

  if frac_part == 0 {
    int_to_string(int_part) ++ ".0"
  } else {
    let mut frac_str = int_to_string(frac_part)
    while frac_str != "" && frac_str[frac_str.length() - 1] == '0' {
      frac_str = frac_str.substring(0, frac_str.length() - 1)
    }
    int_to_string(int_part) ++ "." ++ (if frac_str == "" { "0" } else { frac_str })
  }
}

fn float_to_int(f: Float) -> Int {
  if f < 0.0 { 0 } else {
    let mut result = 0
    while (result as Float) <= f {
      result = result + 1
    }
    result - 1
  }
}

// Module-level metrics instance
let global_metrics = MetricsCollector::new()

/// Get the global metrics collector
pub fn get_global_metrics() -> MetricsCollector {
  global_metrics
}
