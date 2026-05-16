// src/cache/hash.mbt
// Hash utilities for cache key generation
// Reference: lmdeploy-rust-server/src/cache/tokenizer_cache.rs (compute_hash function)

/// Compute hash of text using djb2 algorithm
/// Note: This is a simple hash for development. Production should use SHA-256.
/// For SHA-256, would need FFI to a C library or custom implementation.
pub fn compute_hash(text: String) -> String {
  // djb2 hash algorithm - fast and good distribution for strings
  let mut hash: Int = 5381
  for ch in text {
    hash = ((hash << 5) + hash) + ch.to_int()
  }
  // Convert to hex string (lowercase)
  to_hex(hash)
}

/// Convert integer to hex string
fn to_hex(n: Int) -> String {
  if n == 0 {
    "0"
  } else {
    let hex_chars = "0123456789abcdef"
    let mut result = ""
    let mut num = n
    while num > 0 {
      let idx = num % 16
      result = hex_chars.get(idx) + result
      num = num / 16
    }
    result
  }
}

/// Verify hash consistency
pub fn verify_hash_consistency() -> Bool {
  let h1 = compute_hash("hello")
  let h2 = compute_hash("hello")
  let h3 = compute_hash("world")
  h1 == h2 && h1 != h3
}