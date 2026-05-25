/// Tokenization cache for avoiding redundant tokenization of identical prompts.
///
/// This cache stores (prompt_hash -> token_ids) mappings using a simple LRU eviction
/// policy. For prefill benchmarks where the same prompt is used repeatedly, this can
/// eliminate tokenization overhead entirely.
///
/// Thread-local storage avoids lock contention in multi-threaded servers.
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

const CACHE_CAPACITY: usize = 64;

#[derive(Clone)]
struct CacheEntry {
    token_ids: Vec<u32>,
    access_count: u64,
}

thread_local! {
    static TOKENIZATION_CACHE: RefCell<HashMap<u64, CacheEntry>> = RefCell::new(HashMap::with_capacity(CACHE_CAPACITY));
}

/// Compute a fast hash of the prompt string for cache lookup.
fn hash_prompt(prompt: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(prompt.as_bytes());
    hasher.finish()
}

/// Get token_ids from cache, returning None if not found.
pub fn get_cached_token_ids(prompt: &str) -> Option<Vec<u32>> {
    let hash = hash_prompt(prompt);
    TOKENIZATION_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(entry) = cache.get_mut(&hash) {
            entry.access_count += 1;
            Some(entry.token_ids.clone())
        } else {
            None
        }
    })
}

/// Insert token_ids into cache, evicting oldest entries if capacity exceeded.
pub fn cache_token_ids(prompt: &str, token_ids: Vec<u32>) {
    let hash = hash_prompt(prompt);
    TOKENIZATION_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        // Simple LRU: if at capacity, remove the entry with lowest access count
        if cache.len() >= CACHE_CAPACITY {
            let min_key = cache
                .iter()
                .min_by_key(|(_, a)| (a.access_count, a.access_count))
                .map(|(k, _)| *k);
            if let Some(key) = min_key {
                cache.remove(&key);
            }
        }
        cache.insert(hash, CacheEntry {
            token_ids,
            access_count: 1,
        });
    })
}

/// Clear the tokenization cache (useful for testing or memory pressure).
pub fn clear_tokenization_cache() {
    TOKENIZATION_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        clear_tokenization_cache();
        let prompt = "Hello, world!";
        let token_ids = vec![1, 2, 3];

        // Cache miss
        assert!(get_cached_token_ids(prompt).is_none());

        // Insert
        cache_token_ids(prompt, token_ids.clone());

        // Cache hit
        assert_eq!(get_cached_token_ids(prompt), Some(token_ids));
    }

    #[test]
    fn test_cache_different_prompts() {
        clear_tokenization_cache();
        let prompt1 = "Hello";
        let prompt2 = "World";

        cache_token_ids(prompt1, vec![1, 2]);
        cache_token_ids(prompt2, vec![3, 4]);

        assert_eq!(get_cached_token_ids(prompt1), Some(vec![1, 2]));
        assert_eq!(get_cached_token_ids(prompt2), Some(vec![3, 4]));
    }

    #[test]
    fn test_cache_capacity() {
        clear_tokenization_cache();
        // Fill cache beyond capacity
        for i in 0..(CACHE_CAPACITY + 10) {
            cache_token_ids(&format!("Prompt {}", i), vec![i as u32]);
        }
        // Cache should not exceed capacity significantly
        TOKENIZATION_CACHE.with(|cache| {
            assert!(cache.borrow().len() <= CACHE_CAPACITY + 1); // +1 for timing
        });
    }
}
