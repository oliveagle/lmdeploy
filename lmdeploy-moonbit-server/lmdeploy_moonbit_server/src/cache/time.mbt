// src/cache/time.mbt
// Time utilities for cache TTL support

/// Get current Unix timestamp in seconds
/// Note: This is a placeholder. Production would use FFI or proper time library.
pub fn unix_timestamp_secs() -> Int {
  // Placeholder implementation
  // In production, would call system time via FFI or use runtime function
  0
}

/// Create timestamp with custom offset (for testing)
pub fn unix_timestamp_with_offset(offset_secs: Int) -> Int {
  unix_timestamp_secs() + offset_secs
}