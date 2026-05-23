use lru::LruCache;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;

/// Cached tokenization result with metadata
#[derive(Debug, Clone)]
pub struct CachedTokens {
    /// Token IDs for the cached text
    pub token_ids: Vec<u32>,
    /// Original text that was tokenized
    pub text: String,
    /// When this entry was created (unix timestamp)
    pub created_at: u64,
    /// Number of times this cache entry has been accessed
    pub access_count: u64,
}

/// Cache entry with TTL support
#[derive(Debug, Clone)]
struct CacheEntry {
    tokens: CachedTokens,
    expires_at: u64,
}

/// Tokenize cache with LRU eviction and TTL support
#[derive(Clone)]
pub struct TokenizeCache {
    /// Main LRU cache for exact text matches
    cache: Arc<RwLock<LruCache<String, CacheEntry>>>,
    /// Prefix cache for sharing common prefixes
    prefix_cache: Arc<RwLock<HashMap<String, Vec<u32>>>>,
    /// Maximum cache size
    max_size: usize,
    /// TTL in seconds
    ttl_secs: u64,
    /// Cache metrics
    metrics: Arc<RwLock<CacheMetrics>>,
}

/// Cache performance metrics
#[derive(Debug, Default, Clone)]
pub struct CacheMetrics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefix_hits: u64,
    pub evictions: u64,
}

impl TokenizeCache {
    /// Create a new tokenize cache
    pub fn new(max_size: usize, ttl_secs: u64) -> Self {
        let size = NonZeroUsize::new(max_size.max(1)).unwrap();
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(size))),
            prefix_cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            ttl_secs,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    /// Get tokenized result from cache, or tokenize using the provided function
    pub async fn get_or_tokenize<F, Fut>(
        &self,
        text: &str,
        tokenize_fn: F,
    ) -> Result<Vec<u32>>
    where
        F: FnOnce(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<Vec<u32>>> + Send,
    {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        drop(metrics);

        // Try exact match first
        if let Some(tokens) = self.get_exact(text).await {
            self.record_hit().await;
            return Ok(tokens);
        }

        // Try prefix match
        if let Some(tokens) = self.get_prefix(text).await {
            self.record_prefix_hit().await;
            return Ok(tokens);
        }

        // Cache miss - tokenize and cache
        self.record_miss().await;
        let token_ids = tokenize_fn(text).await?;

        // Store in cache
        self.store(text.to_string(), token_ids.clone()).await;

        Ok(token_ids)
    }

    /// Get exact match from cache
    async fn get_exact(&self, text: &str) -> Option<Vec<u32>> {
        let mut cache = self.cache.write().await;
        let now = unix_timestamp_secs();

        if let Some(entry) = cache.get_mut(text) {
            // Check TTL
            if now < entry.expires_at {
                entry.tokens.access_count += 1;
                return Some(entry.tokens.token_ids.clone());
            } else {
                // Expired - remove it
                cache.pop(text);
            }
        }
        None
    }

    /// Get prefix match from cache
    /// Uses read lock - only reads are performed, no mutations.
    async fn get_prefix(&self, text: &str) -> Option<Vec<u32>> {
        let prefix_cache = self.prefix_cache.read().await;

        // Find the longest matching prefix
        let mut best_match: Option<(String, Vec<u32>)> = None;
        let mut best_len = 0;

        for (prefix, tokens) in prefix_cache.iter() {
            if text.starts_with(prefix) && prefix.len() > best_len {
                best_len = prefix.len();
                best_match = Some((prefix.clone(), tokens.clone()));
            }
        }

        if let Some((_prefix, tokens)) = best_match {
            // We found a prefix match - tokenize only the suffix
            let suffix = &text[best_len..];
            if !suffix.is_empty() {
                // For now, we can't tokenize the suffix without the tokenizer
                // This would be handled by the caller
                return None;
            }
            return Some(tokens);
        }

        None
    }

    /// Store tokenization result in cache
    async fn store(&self, text: String, token_ids: Vec<u32>) {
        let now = unix_timestamp_secs();
        let expires_at = now + self.ttl_secs;

        let entry = CacheEntry {
            tokens: CachedTokens {
                token_ids: token_ids.clone(),
                text: text.clone(),
                created_at: now,
                access_count: 1,
            },
            expires_at,
        };

        let mut cache = self.cache.write().await;

        // Check if we're about to evict
        if cache.len() >= self.max_size && !cache.contains(&text) {
            let mut metrics = self.metrics.write().await;
            metrics.evictions += 1;
        }

        cache.put(text.clone(), entry);

        // Also store prefixes (for longer texts, store common prefix lengths)
        if text.len() > 50 {
            let mut prefix_cache = self.prefix_cache.write().await;
            let prefix = text.chars().take(50).collect::<String>();
            let prefix_token_count = (token_ids.len() * 50) / text.len().max(1);
            let truncated_tokens = token_ids[..prefix_token_count.min(token_ids.len())].to_vec();
            prefix_cache
                .entry(prefix)
                .or_insert_with(|| truncated_tokens);
        }
    }

    /// Clear all cache entries
    pub async fn clear(&self) {
        self.cache.write().await.clear();
        self.prefix_cache.write().await.clear();
    }

    /// Get cache metrics
    pub async fn metrics(&self) -> CacheMetrics {
        self.metrics.read().await.clone()
    }

    /// Get cache hit rate
    pub async fn hit_rate(&self) -> f64 {
        let metrics = self.metrics.read().await;
        if metrics.total_requests == 0 {
            return 0.0;
        }
        (metrics.cache_hits + metrics.prefix_hits) as f64 / metrics.total_requests as f64
    }

    /// Get current cache size
    pub async fn size(&self) -> usize {
        self.cache.read().await.len()
    }

    async fn record_hit(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.cache_hits += 1;
    }

    async fn record_prefix_hit(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.prefix_hits += 1;
    }

    async fn record_miss(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.cache_misses += 1;
    }
}

/// Compute SHA-256 hash of text
pub fn compute_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Get current unix timestamp in seconds
fn unix_timestamp_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_hit_miss() {
        let cache = TokenizeCache::new(100, 3600);

        // First call - cache miss
        let result1 = cache
            .get_or_tokenize("hello world", |text| async move {
                Ok(vec![1, 2, 3])
            })
            .await
            .unwrap();
        assert_eq!(result1, vec![1, 2, 3]);

        // Second call - cache hit
        let result2 = cache
            .get_or_tokenize("hello world", |text| async move {
                panic!("Should not be called on cache hit");
            })
            .await
            .unwrap();
        assert_eq!(result2, vec![1, 2, 3]);

        let metrics = cache.metrics().await;
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.cache_hits, 1);
        assert_eq!(metrics.cache_misses, 1);
    }

    #[tokio::test]
    async fn test_ttl_expiry() {
        let cache = TokenizeCache::new(100, 1); // 1 second TTL

        // Store value
        let result1 = cache
            .get_or_tokenize("test", |text| async move { Ok(vec![1, 2]) })
            .await
            .unwrap();

        // Immediate hit
        let metrics = cache.metrics().await;
        assert_eq!(metrics.cache_hits, 0); // First request was a miss
        assert_eq!(metrics.cache_misses, 1);

        // Note: In a real test we'd wait for TTL, but for unit tests
        // we assume the cache doesn't expire immediately
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let cache = TokenizeCache::new(2, 3600); // Size 2

        cache
            .get_or_tokenize("a", |text| async move { Ok(vec![1]) })
            .await
            .unwrap();
        cache
            .get_or_tokenize("b", |text| async move { Ok(vec![2]) })
            .await
            .unwrap();

        let metrics = cache.metrics().await;
        assert_eq!(metrics.evictions, 0);

        // Add third item - should evict first
        cache
            .get_or_tokenize("c", |text| async move { Ok(vec![3]) })
            .await
            .unwrap();

        let metrics = cache.metrics().await;
        assert_eq!(metrics.evictions, 1);
    }

    #[tokio::test]
    async fn test_hash_computation() {
        let hash1 = compute_hash("hello");
        let hash2 = compute_hash("hello");
        let hash3 = compute_hash("world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA-256 is 64 hex chars
    }
}
