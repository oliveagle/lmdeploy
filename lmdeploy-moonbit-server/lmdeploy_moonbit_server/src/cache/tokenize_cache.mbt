// src/cache/tokenize_cache.mbt
// Tokenization cache with LRU eviction and TTL support
// Reference: lmdeploy-rust-server/src/cache/tokenizer_cache.rs

/// Cached tokenization result with metadata
pub struct CachedTokens {
  token_ids: Array[Int]
  text: String
  created_at: Int
  access_count: Int
}

/// Cache entry with TTL expiration
struct CacheEntry {
  tokens: CachedTokens
  expires_at: Int
}

/// Cache performance metrics
pub struct CacheMetrics {
  total_requests: Int
  cache_hits: Int
  cache_misses: Int
  prefix_hits: Int
  evictions: Int
}

pub fn CacheMetrics::default() -> CacheMetrics {
  {
    total_requests: 0,
    cache_hits: 0,
    cache_misses: 0,
    prefix_hits: 0,
    evictions: 0,
  }
}

/// LRU doubly-linked list node for ordering tracking
struct LruNode {
  key: String
  prev: Int
  next: Int
}

pub struct TokenizeCache {
  cache: Map[String, CacheEntry]
  prefix_cache: Map[String, Array[Int]]
  max_size: Int
  ttl_secs: Int
  metrics: CacheMetrics
  // LRU tracking: ordered list of keys (oldest first)
  lru_order: Array[String]
}

pub fn TokenizeCache::new(max_size: Int, ttl_secs: Int) -> TokenizeCache {
  {
    cache: Map::new(),
    prefix_cache: Map::new(),
    max_size,
    ttl_secs,
    metrics: CacheMetrics::default(),
    lru_order: [],
  }
}

/// Get tokenized result from cache, or tokenize using the provided function
pub fn TokenizeCache::get_or_tokenize(
  self: TokenizeCache,
  text: String,
  tokenize_fn: Fn(String) -> Result[Array[Int], String],
) -> (TokenizeCache, Result[Array[Int], String]) {
  self = self.record_total_request()

  // Try exact match first
  let (self, exact_result) = self.get_exact(text)
  match exact_result {
    Some(tokens) => {
      let self = self.record_hit()
      (self, Ok(tokens))
    }
    None => {
      // Try prefix match
      let (self, prefix_result) = self.get_prefix(text)
      match prefix_result {
        Some(tokens) => {
          let self = self.record_prefix_hit()
          (self, Ok(tokens))
        }
        None => {
          // Cache miss - tokenize and cache
          let self = self.record_miss()
          let token_ids = tokenize_fn(text)
          match token_ids {
            Ok(ids) => {
              let self = self.store(text, ids)
              (self, Ok(ids))
            }
            Err(e) => (self, Err(e)),
          }
        }
      }
    }
  }
}

/// Get exact match from cache
fn TokenizeCache::get_exact(self: TokenizeCache, text: String) -> (TokenizeCache, Option[Array[Int]]) {
  let now = unix_timestamp_secs()

  match self.cache.get(text) {
    Some(entry) => {
      if now < entry.expires_at {
        // Hit - update access count and move to MRU position
        let mut updated_entry = entry
        updated_entry.tokens = {
          token_ids: updated_entry.tokens.token_ids,
          text: updated_entry.tokens.text,
          created_at: updated_entry.tokens.created_at,
          access_count: updated_entry.tokens.access_count + 1,
        }
        self = {
          cache: self.cache.update(text, updated_entry),
          prefix_cache: self.prefix_cache,
          max_size: self.max_size,
          ttl_secs: self.ttl_secs,
          metrics: self.metrics,
          lru_order: move_to_mru(self.lru_order, text),
        }
        (self, Some(updated_entry.tokens.token_ids))
      } else {
        // Expired - remove it
        self = {
          cache: self.cache.remove(text),
          prefix_cache: self.prefix_cache,
          max_size: self.max_size,
          ttl_secs: self.ttl_secs,
          metrics: self.metrics,
          lru_order: self.lru_order.filter(fn (key) { key != text }),
        }
        (self, None)
      }
    }
    None => (self, None),
  }
}

/// Get prefix match from cache
fn TokenizeCache::get_prefix(self: TokenizeCache, text: String) -> (TokenizeCache, Option[Array[Int]]) {
  let mut best_match: Option[Array[Int]] = None
  let mut best_len = 0

  for (prefix, tokens) in self.prefix_cache {
    if text.starts_with(prefix) && prefix.length() > best_len {
      best_len = prefix.length()
      best_match = Some(tokens)
    }
  }

  match best_match {
    Some(tokens) => (self, Some(tokens)),
    None => (self, None),
  }
}

/// Store tokenization result in cache
fn TokenizeCache::store(self: TokenizeCache, text: String, token_ids: Array[Int]) -> TokenizeCache {
  let now = unix_timestamp_secs()
  let expires_at = now + self.ttl_secs

  let entry: CacheEntry = {
    tokens: {
      token_ids,
      text,
      created_at: now,
      access_count: 1,
    },
    expires_at,
  }

  // Check if we need to evict
  let mut self = self
  if self.cache.length() >= self.max_size && !self.cache.contains(text) {
    // Evict LRU entry (first in lru_order)
    if self.lru_order.length() > 0 {
      let lru_key = self.lru_order[0]
      self = {
        cache: self.cache.remove(lru_key),
        prefix_cache: self.prefix_cache,
        max_size: self.max_size,
        ttl_secs: self.ttl_secs,
        metrics: record_eviction(self.metrics),
        lru_order: self.lru_order.slice(1, self.lru_order.length()),
      }
    }
  }

  // Insert into main cache
  self = {
    cache: self.cache.insert(text, entry),
    prefix_cache: self.prefix_cache,
    max_size: self.max_size,
    ttl_secs: self.ttl_secs,
    metrics: self.metrics,
    lru_order: self.lru_order.push(text),
  }

  // Also store prefixes for longer texts
  if text.length() > 50 {
    let prefix = text.substring(0, min(50, text.length()))
    let prefix_token_count = (token_ids.length() * 50) / max(1, text.length())
    let truncated_tokens = token_ids.slice(0, min(prefix_token_count, token_ids.length()))
    if !self.prefix_cache.contains(prefix) {
      self = {
        cache: self.cache,
        prefix_cache: self.prefix_cache.insert(prefix, truncated_tokens),
        max_size: self.max_size,
        ttl_secs: self.ttl_secs,
        metrics: self.metrics,
        lru_order: self.lru_order,
      }
    }
  }

  self
}

/// Clear all cache entries
pub fn TokenizeCache::clear(self: TokenizeCache) -> TokenizeCache {
  {
    cache: Map::new(),
    prefix_cache: Map::new(),
    max_size: self.max_size,
    ttl_secs: self.ttl_secs,
    metrics: self.metrics,
    lru_order: [],
  }
}

/// Get cache hit rate as a percentage
pub fn TokenizeCache::hit_rate(self: TokenizeCache) -> Float {
  let total = self.metrics.cache_hits + self.metrics.prefix_hits
  if self.metrics.total_requests == 0 {
    0.0
  } else {
    to_float(total) /. to_float(self.metrics.total_requests)
  }
}

/// Get current cache size
pub fn TokenizeCache::size(self: TokenizeCache) -> Int {
  self.cache.length()
}

/// Get cache metrics
pub fn TokenizeCache::metrics(self: TokenizeCache) -> CacheMetrics {
  self.metrics
}

fn TokenizeCache::record_total_request(self: TokenizeCache) -> TokenizeCache {
  {
    cache: self.cache,
    prefix_cache: self.prefix_cache,
    max_size: self.max_size,
    ttl_secs: self.ttl_secs,
    metrics: record_total_request(self.metrics),
    lru_order: self.lru_order,
  }
}

fn TokenizeCache::record_hit(self: TokenizeCache) -> TokenizeCache {
  {
    cache: self.cache,
    prefix_cache: self.prefix_cache,
    max_size: self.max_size,
    ttl_secs: self.ttl_secs,
    metrics: record_hit(self.metrics),
    lru_order: self.lru_order,
  }
}

fn TokenizeCache::record_miss(self: TokenizeCache) -> TokenizeCache {
  {
    cache: self.cache,
    prefix_cache: self.prefix_cache,
    max_size: self.max_size,
    ttl_secs: self.ttl_secs,
    metrics: record_miss(self.metrics),
    lru_order: self.lru_order,
  }
}

fn TokenizeCache::record_prefix_hit(self: TokenizeCache) -> TokenizeCache {
  {
    cache: self.cache,
    prefix_cache: self.prefix_cache,
    max_size: self.max_size,
    ttl_secs: self.ttl_secs,
    metrics: record_prefix_hit(self.metrics),
    lru_order: self.lru_order,
  }
}

/// Move a key to the most-recently-used position
fn move_to_mru(order: Array[String], key: String) -> Array[String] {
  let filtered = order.filter(fn (k) { k != key })
  filtered.push(key)
}

fn record_eviction(metrics: CacheMetrics) -> CacheMetrics {
  {
    total_requests: metrics.total_requests,
    cache_hits: metrics.cache_hits,
    cache_misses: metrics.cache_misses,
    prefix_hits: metrics.prefix_hits,
    evictions: metrics.evictions + 1,
  }
}

fn record_total_request(metrics: CacheMetrics) -> CacheMetrics {
  {
    total_requests: metrics.total_requests + 1,
    cache_hits: metrics.cache_hits,
    cache_misses: metrics.cache_misses,
    prefix_hits: metrics.prefix_hits,
    evictions: metrics.evictions,
  }
}

fn record_hit(metrics: CacheMetrics) -> CacheMetrics {
  {
    total_requests: metrics.total_requests,
    cache_hits: metrics.cache_hits + 1,
    cache_misses: metrics.cache_misses,
    prefix_hits: metrics.prefix_hits,
    evictions: metrics.evictions,
  }
}

fn record_miss(metrics: CacheMetrics) -> CacheMetrics {
  {
    total_requests: metrics.total_requests,
    cache_hits: metrics.cache_hits,
    cache_misses: metrics.cache_misses + 1,
    prefix_hits: metrics.prefix_hits,
    evictions: metrics.evictions,
  }
}

fn record_prefix_hit(metrics: CacheMetrics) -> CacheMetrics {
  {
    total_requests: metrics.total_requests,
    cache_hits: metrics.cache_hits,
    cache_misses: metrics.cache_misses,
    prefix_hits: metrics.prefix_hits + 1,
    evictions: metrics.evictions,
  }
}
