// src/cache/hash.mbt
// Hash utilities for cache key generation

/// Convert integer to hex string
fn int_to_hex(n: Int) -> String {
  let mut result = ""
  let mut num = n
  if num < 0 {
    num = -num
  }
  if num == 0 {
    return "0"
  }
  while num > 0 {
    let digit = num % 16
    let ch = digit_to_hex_char(digit)
    result = ch + result
    num = num / 16
  }
  result
}

/// Convert a single digit to hex char
fn digit_to_hex_char(digit: Int) -> String {
  match digit {
    0 => "0"
    1 => "1"
    2 => "2"
    3 => "3"
    4 => "4"
    5 => "5"
    6 => "6"
    7 => "7"
    8 => "8"
    9 => "9"
    10 => "a"
    11 => "b"
    12 => "c"
    13 => "d"
    14 => "e"
    _ => "f"
  }
}

/// Compute hash of text using djb2 algorithm
/// Note: This is a simple hash for development. Production should use SHA-256.
pub fn compute_hash(text: String) -> String {
  // djb2 hash algorithm - fast and good distribution for strings
  let mut hash: Int = 5381
  for ch in text {
    hash = ((hash << 5) + hash) + ch.to_int()
  }
  // Convert to hex string (lowercase)
  int_to_hex(hash)
}

/// Verify hash consistency
pub fn verify_hash_consistency() -> Bool {
  let h1 = compute_hash("hello")
  let h2 = compute_hash("hello")
  let h3 = compute_hash("world")
  h1 == h2 && h1 != h3
}