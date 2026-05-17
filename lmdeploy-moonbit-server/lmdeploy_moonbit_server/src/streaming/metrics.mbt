// src/streaming/metrics.mbt
// Streaming metrics for tracking first token latency and chunk delivery
// Reference: lmdeploy-rust-server/src/metrics.rs (StreamMetrics)

/// Streaming metrics snapshot (immutable, functional style)
pub struct StreamingMetrics {
  /// Total streams initiated
  total_streams: Int
  /// Cumulative first token latency in microseconds (sum for averaging)
  cumulative_first_token_us: Int
  /// Total chunks sent across all streams
  total_chunks_sent: Int
  /// Total stream timeouts
  stream_timeouts: Int
}

pub fn StreamingMetrics::new() -> StreamingMetrics {
  {
    total_streams: 0,
    cumulative_first_token_us: 0,
    total_chunks_sent: 0,
    stream_timeouts: 0,
  }
}

/// Record a new stream start with first token latency in microseconds
pub fn StreamingMetrics::record_stream_start(self : StreamingMetrics, first_token_us: Int) -> StreamingMetrics {
  {
    total_streams: self.total_streams + 1,
    cumulative_first_token_us: self.cumulative_first_token_us + first_token_us,
    total_chunks_sent: self.total_chunks_sent,
    stream_timeouts: self.stream_timeouts,
  }
}

/// Record a chunk sent
pub fn StreamingMetrics::record_chunk(self : StreamingMetrics) -> StreamingMetrics {
  {
    total_streams: self.total_streams,
    cumulative_first_token_us: self.cumulative_first_token_us,
    total_chunks_sent: self.total_chunks_sent + 1,
    stream_timeouts: self.stream_timeouts,
  }
}

/// Record a stream timeout
pub fn StreamingMetrics::record_timeout(self : StreamingMetrics) -> StreamingMetrics {
  {
    total_streams: self.total_streams,
    cumulative_first_token_us: self.cumulative_first_token_us,
    total_chunks_sent: self.total_chunks_sent,
    stream_timeouts: self.stream_timeouts + 1,
  }
}

/// Get average first token latency in microseconds
pub fn StreamingMetrics::avg_first_token_latency_us(self : StreamingMetrics) -> Float {
  if self.total_streams > 0 {
    self.cumulative_first_token_us as Float / self.total_streams as Float
  } else {
    0.0
  }
}

/// Convert metrics to Prometheus format
pub fn StreamingMetrics::to_prometheus(self : StreamingMetrics) -> String {
  "# HELP stream_total Total number of streams initiated\n" +
  "# TYPE stream_total counter\n" +
  "stream_total " + int_to_string(self.total_streams) + "\n" +
  "# HELP stream_first_token_latency_avg_us Average first token latency in microseconds\n" +
  "# TYPE stream_first_token_latency_avg_us gauge\n" +
  "stream_first_token_latency_avg_us " + int_to_string(self.avg_first_token_latency_us() as Int) + "\n" +
  "# HELP stream_chunks_total Total number of chunks sent\n" +
  "# TYPE stream_chunks_total counter\n" +
  "stream_chunks_total " + int_to_string(self.total_chunks_sent) + "\n" +
  "# HELP stream_timeouts_total Total number of stream timeouts\n" +
  "# TYPE stream_timeouts_total counter\n" +
  "stream_timeouts_total " + int_to_string(self.stream_timeouts)
}

/// Convert metrics to JSON
pub fn StreamingMetrics::to_json(self : StreamingMetrics) -> String {
  "{
  \"total_streams\": " + int_to_string(self.total_streams) + ",
  \"avg_first_token_latency_us\": " + float_to_string(self.avg_first_token_latency_us()) + ",
  \"total_chunks_sent\": " + int_to_string(self.total_chunks_sent) + ",
  \"stream_timeouts\": " + int_to_string(self.stream_timeouts) + "
}"
}

/// Helper: integer to string
fn int_to_string(i: Int) -> String {
  if i < 0 { "-" ++ uint_to_string(-i) } else { uint_to_string(i) }
}

fn uint_to_string(n: Int) -> String {
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

fn digit_to_char(d: Int) -> String {
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

/// Helper: float to string
fn float_to_string(f: Float) -> String {
  if f == 0.0 { "0.0" }
  else if f == 1.0 { "1.0" }
  else if f == -1.0 { "-1.0" }
  else if f == 0.5 { "0.5" }
  else if f == 100.0 { "100.0" }
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

fn float_to_int(f: Float) -> Int {
  if f < 0.0 { 0 } else {
    let mut result = 0
    while (result as Float) <= f {
      result = result + 1
    }
    result - 1
  }
}
