// src/cache/time.mbt
// Time utilities for cache TTL support

/// Get current Unix timestamp in seconds
pub fn unix_timestamp_secs() -> Int {
  // In WASI environment, use clock_time_get
  // For non-WASI or testing, return 0
  get_system_time_secs()
}

/// Create timestamp with custom offset (for testing)
pub fn unix_timestamp_with_offset(offset_secs: Int) -> Int {
  get_system_time_secs() + offset_secs
}

/// Get system time in seconds using WASI clock_time_get if available
fn get_system_time_secs() -> Int {
  // WASI realtime clock returns nanoseconds
  // clock_time_get(clockid=0=REALTIME, precision=1_000_000_000)
  let result = wasi_clock_time_get(0, 1_000_000_000)
  // Convert nanoseconds to seconds
  result / 1_000_000_000
}

/// WASI clock_time_get FFI binding
/// Returns nanoseconds since Unix epoch for REALTIME clock
fn wasi_clock_time_get(clock_id: Int, precision: Int) -> Int64 {
  // In production:
  // extern fn wasi_snapshot_preview1::clock_time_get(
  //   clock_id: i32,
  //   precision: i64,
  //   result: *mut i64
  // ) -> i32
  //
  // For now, return 0 as WASI is not linked at compile time
  // The actual FFI binding would be provided by the WASI runtime
  0L
}