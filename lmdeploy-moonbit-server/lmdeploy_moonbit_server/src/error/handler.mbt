// src/error/handler.mbt
// Unified error handler: converts errors to HTTP responses, integrates rate limiting
// Reference: Python api_server.py request throttling, Rust rate limiter patterns

use username/lmdeploy_moonbit_server/error
use username/lmdeploy_moonbit_server/ratelimit

/// Error handler manages rate limiting and error response formatting
pub struct ErrorHandler {
  rate_limiter: ratelimit.RateLimiter
  request_timeout_secs: Int
  max_requests_per_window: Int
  window_secs: Int
}

pub fn ErrorHandler::new(max_requests: Int, window_secs: Int, timeout_secs: Int) -> ErrorHandler {
  {
    rate_limiter: ratelimit.RateLimiter::new(max_requests, window_secs),
    request_timeout_secs: timeout_secs,
    max_requests_per_window: max_requests,
    window_secs: window_secs,
  }
}

pub fn ErrorHandler::default() -> ErrorHandler {
  {
    rate_limiter: ratelimit.RateLimiter::default(),
    request_timeout_secs: 300,
    max_requests_per_window: ratelimit.DEFAULT_MAX_REQUESTS,
    window_secs: ratelimit.DEFAULT_WINDOW_SECS,
  }
}

/// Check rate limit for a client, return error message if exceeded
pub fn ErrorHandler::check_rate_limit(
  self: ErrorHandler,
  client_id: String,
) -> (ErrorHandler, Option[String]) {
  let (updated_limiter, allowed) = self.rate_limiter.allow_request(client_id)
  let updated_self = {
    rate_limiter: updated_limiter,
    request_timeout_secs: self.request_timeout_secs,
    max_requests_per_window: self.max_requests_per_window,
    window_secs: self.window_secs,
  }
  if allowed {
    (updated_self, None)
  } else {
    let retry_after = self.window_secs
    let msg = "Too many requests. Please retry after " ++ int_to_string(retry_after) ++ " seconds."
    (updated_self, Some(build_error_json(429, "rate_limit_exceeded", msg)))
  }
}

/// Get rate limit exceeded response body
pub fn ErrorHandler::rate_limit_exceeded_response() -> String {
  build_error_json(
    429,
    "rate_limit_exceeded",
    "Too many requests. Please retry later.",
  )
}

/// Get timeout error response body
pub fn ErrorHandler::timeout_response() -> String {
  let msg = "Request timed out. Please try again with a shorter request."
  build_error_json(408, "timeout_error", msg)
}

/// Get internal server error response body
pub fn ErrorHandler::internal_error_response(message: String) -> String {
  build_error_json(500, "internal_error", message)
}

/// Get service unavailable response body
pub fn ErrorHandler::service_unavailable_response() -> String {
  build_error_json(
    503,
    "service_unavailable",
    "Service is temporarily unavailable. Please try again later.",
  )
}

/// Get out of memory error response body
pub fn ErrorHandler::oom_response(model: String) -> String {
  let msg = "Out of memory while loading model: " ++ model ++ ". Please retry with a smaller model or reduce batch size."
  build_error_json(507, "insufficient_storage", msg)
}

/// Get bad request error response body
pub fn ErrorHandler::bad_request_response(message: String) -> String {
  build_error_json(400, "invalid_request_error", message)
}

/// Get not found error response body
pub fn ErrorHandler::not_found_response(message: String) -> String {
  build_error_json(404, "not_found", message)
}

/// Convert LmDeployError to response JSON
pub fn ErrorHandler::from_lm_error(error: LmDeployError) -> String {
  error.to_json()
}

/// Periodic cleanup of expired rate limit entries
pub fn ErrorHandler::cleanup(self: ErrorHandler) -> ErrorHandler {
  {
    rate_limiter: self.rate_limiter.cleanup(),
    request_timeout_secs: self.request_timeout_secs,
    max_requests_per_window: self.max_requests_per_window,
    window_secs: self.window_secs,
  }
}

/// Get request timeout in seconds
pub fn ErrorHandler::timeout_secs(self: ErrorHandler) -> Int {
  self.request_timeout_secs
}

/// Get max requests per window
pub fn ErrorHandler::max_requests(self: ErrorHandler) -> Int {
  self.max_requests_per_window
}

/// Get window size in seconds
pub fn ErrorHandler::window_secs(self: ErrorHandler) -> Int {
  self.window_secs
}

/// Get current client count
pub fn ErrorHandler::client_count(self: ErrorHandler) -> Int {
  self.rate_limiter.client_count()
}

/// Export rate limiting metrics for Prometheus
pub fn ErrorHandler::to_prometheus(self: ErrorHandler) -> String {
  "# HELP lmdeploy_ratelimit_clients_current Current number of rate-limited clients\n"
  ++ "# TYPE lmdeploy_ratelimit_clients_current gauge\n"
  ++ "lmdeploy_ratelimit_clients_current "
  ++ int_to_string(self.client_count())
  ++ "\n"
  ++ "# HELP lmdeploy_ratelimit_max_requests Max requests allowed per window\n"
  ++ "# TYPE lmdeploy_ratelimit_max_requests gauge\n"
  ++ "lmdeploy_ratelimit_max_requests "
  ++ int_to_string(self.max_requests_per_window)
  ++ "\n"
}