// src/batch/batch.mbt
// Batch processing for LMDeploy MoonBit API
// Reference: lmdeploy-rust-server/src/batch.rs

/// Batch item for processing
pub struct BatchItem {
  id: String
  request_type: RequestType
  payload: String
  timestamp: Int
  priority: Int
}

pub fn BatchItem::new(request_type: RequestType, payload: String) -> BatchItem {
  {
    id: generate_batch_item_id(),
    request_type,
    payload,
    timestamp: current_timestamp_secs(),
    priority: 0,
  }
}

/// Request type enum
pub enum RequestType {
  ChatCompletion
  Completion
  Embedding
  Encode
}

pub fn RequestType::to_string(self : RequestType) -> String {
  match self {
    ChatCompletion => "chat_completion"
    Completion => "completion"
    Embedding => "embedding"
    Encode => "encode"
  }
}

/// Batch configuration
pub struct BatchConfig {
  max_size: Int
  timeout_ms: Int
  enable_batching: Bool
}

pub fn BatchConfig::default() -> BatchConfig {
  {
    max_size: 8,
    timeout_ms: 50,
    enable_batching: true,
  }
}

/// Batch statistics
pub struct BatchStats {
  total_batches: Int
  total_requests: Int
  avg_batch_size: Float
  max_batch_size: Int
  min_batch_size: Int
  total_latency_ms: Int
}

pub fn BatchStats::default() -> BatchStats {
  {
    total_batches: 0,
    total_requests: 0,
    avg_batch_size: 0.0,
    max_batch_size: 0,
    min_batch_size: 0,
    total_latency_ms: 0,
  }
}

/// Batch processor with request batching
pub struct BatchProcessor {
  config: BatchConfig
  pending_queue: Array[BatchItem]
  stats: BatchStats
  last_batch_time: Int
}

pub fn BatchProcessor::new(config: BatchConfig) -> BatchProcessor {
  {
    config,
    pending_queue: [],
    stats: BatchStats::default(),
    last_batch_time: 0,
  }
}

/// Add item to batch queue
pub fn BatchProcessor::add_item(
  self: BatchProcessor,
  request_type: RequestType,
  payload: String,
) -> BatchProcessor {
  let item = BatchItem::new(request_type, payload)
  {
    config: self.config,
    pending_queue: self.pending_queue.push(item),
    stats: self.stats,
    last_batch_time: self.last_batch_time,
  }
}

/// Process batch when ready
pub fn BatchProcessor::maybe_process_batch(
  self: BatchProcessor,
  process_fn: Fn(Array[BatchItem]) -> Array[String],
) -> Option[(BatchProcessor, Array[String])] {
  let now = current_timestamp_secs()
  let queue_size = self.pending_queue.length()

  // Process if batch is full or timeout reached
  if queue_size == 0 {
    return None
  }

  let should_process = queue_size >= self.config.max_size ||
    (now - self.last_batch_time) * 1000 >= self.config.timeout_ms

  if !should_process {
    return None
  }

  // Take items from queue
  let items = self.pending_queue
  let results = process_fn(items)

  // Update stats
  let new_stats = {
    total_batches: self.stats.total_batches + 1,
    total_requests: self.stats.total_requests + queue_size,
    avg_batch_size: float(self.stats.total_requests + queue_size) /. float(self.stats.total_batches + 1),
    max_batch_size: if queue_size > self.stats.max_batch_size { queue_size } else { self.stats.max_batch_size },
    min_batch_size: if queue_size < self.stats.min_batch_size || self.stats.min_batch_size == 0 { queue_size } else { self.stats.min_batch_size },
    total_latency_ms: self.stats.total_latency_ms,
  }

  Some(({
    config: self.config,
    pending_queue: [],
    stats: new_stats,
    last_batch_time: now,
  }, results))
}

/// Force process all pending items
pub fn BatchProcessor::force_process(
  self: BatchProcessor,
  process_fn: Fn(Array[BatchItem]) -> Array[String],
) -> (BatchProcessor, Array[String]) {
  let items = self.pending_queue
  let results = if items.length() == 0 {
    []
  } else {
    process_fn(items)
  }

  let queue_size = items.length()
  let now = current_timestamp_secs()

  let new_stats = {
    total_batches: self.stats.total_batches + 1,
    total_requests: self.stats.total_requests + queue_size,
    avg_batch_size: if self.stats.total_batches + 1 > 0 {
      float(self.stats.total_requests + queue_size) /. float(self.stats.total_batches + 1)
    } else {
      float(queue_size)
    },
    max_batch_size: if queue_size > self.stats.max_batch_size { queue_size } else { self.stats.max_batch_size },
    min_batch_size: if queue_size < self.stats.min_batch_size || self.stats.min_batch_size == 0 { queue_size } else { self.stats.min_batch_size },
    total_latency_ms: self.stats.total_latency_ms,
  }

  ({
    config: self.config,
    pending_queue: [],
    stats: new_stats,
    last_batch_time: now,
  }, results)
}

/// Get current queue size
pub fn BatchProcessor::queue_size(self : BatchProcessor) -> Int {
  self.pending_queue.length()
}

/// Get batch statistics
pub fn BatchProcessor::stats(self : BatchProcessor) -> BatchStats {
  self.stats
}

/// Check if batching is enabled
pub fn BatchProcessor::is_enabled(self : BatchProcessor) -> Bool {
  self.config.enable_batching
}

// Helper functions
fn generate_batch_item_id() -> String {
  "batch-" ++ int_to_string(current_timestamp_ms())
}

fn current_timestamp_secs() -> Int {
  // TODO: Implement FFI to time()
  0
}

fn current_timestamp_ms() -> Int {
  current_timestamp_secs() * 1000
}

fn float(i: Int) -> Float {
  // TODO: Proper float conversion
  0.0
}