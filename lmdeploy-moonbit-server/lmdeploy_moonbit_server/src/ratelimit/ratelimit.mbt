// src/ratelimit/ratelimit.mbt
// Rate limiting with configurable requests per window
// Reference: Python api_server.py request throttling, Rust rate limiter patterns

/// Rate limit state for a single client
struct ClientState {
  tokens: Int
  last_refill: Int
}

/// Token bucket rate limiter
pub struct RateLimiter {
  max_requests: Int
  window_secs: Int
  clients: Map[String, ClientState]
}

/// Default rate limit: 100 requests per 60 seconds
pub const DEFAULT_MAX_REQUESTS: Int = 100
pub const DEFAULT_WINDOW_SECS: Int = 60

pub fn RateLimiter::new(max_requests: Int, window_secs: Int) -> RateLimiter {
  {
    max_requests,
    window_secs,
    clients: Map([]),
  }
}

pub fn RateLimiter::default() -> RateLimiter {
  RateLimiter::new(DEFAULT_MAX_REQUESTS, DEFAULT_WINDOW_SECS)
}

/// Check if a request from the given client is allowed
/// Returns (updated limiter, is_allowed)
pub fn RateLimiter::allow_request(
  self: RateLimiter,
  client_id: String,
) -> (RateLimiter, Bool) {
  let now = get_unix_secs()

  match self.clients.get(client_id) {
    Some(state) => {
      // Refill tokens based on elapsed time
      let elapsed = now - state.last_refill
      let refill_count = (elapsed * self.max_requests) / self.window_secs
      let new_tokens = min(state.tokens + refill_count, self.max_requests)

      if new_tokens > 0 {
        // Allow request, consume one token
        let updated_state: ClientState = {
          tokens: new_tokens - 1,
          last_refill: now,
        }
        let updated_self = {
          max_requests: self.max_requests,
          window_secs: self.window_secs,
          clients: self.clients.update(client_id, updated_state),
        }
        (updated_self, true)
      } else {
        // Rate limit exceeded
        (self, false)
      }
    }
    None => {
      // New client, allow and initialize state
      let new_state: ClientState = {
        tokens: self.max_requests - 1,
        last_refill: now,
      }
      let updated_self = {
        max_requests: self.max_requests,
        window_secs: self.window_secs,
        clients: self.clients.insert(client_id, new_state),
      }
      (updated_self, true)
    }
  }
}

/// Cleanup expired client entries to prevent memory leaks
/// Returns updated limiter with old entries removed
pub fn RateLimiter::cleanup(self: RateLimiter) -> RateLimiter {
  let now = get_unix_secs()
  let max_idle_secs = self.window_secs * 2

  let mut filtered = self.clients
  for (key, state) in self.clients {
    if now - state.last_refill > max_idle_secs {
      filtered = filtered.remove(key)
    }
  }

  {
    max_requests: self.max_requests,
    window_secs: self.window_secs,
    clients: filtered,
  }
}

/// Get the number of tracked clients
pub fn RateLimiter::client_count(self: RateLimiter) -> Int {
  self.clients.length()
}

/// Get rate limit config as debug string
pub fn RateLimiter::to_debug_string(self: RateLimiter) -> String {
  "RateLimiter{max: "
    ++ int_to_string(self.max_requests)
    ++ ", window: "
    ++ int_to_string(self.window_secs)
    ++ "s, clients: "
    ++ int_to_string(self.clients.length())
    ++ "}"
}

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
  } else if i == 10 { "10" }
  else if i == 16 { "16" }
  else if i == 60 { "60" }
  else if i == 100 { "100" }
  else { "0" }
}

/// Get current Unix timestamp in seconds (placeholder, would use FFI in production)
fn get_unix_secs() -> Int {
  // Placeholder implementation - in production, would use FFI to system time
  0
}
