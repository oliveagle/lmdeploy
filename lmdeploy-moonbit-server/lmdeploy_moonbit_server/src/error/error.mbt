// src/error/error.mbt
// Error types and handling for LMDeploy MoonBit server
// Reference: lmdeploy-rust-server/src/error.rs

/// LMDeploy error types
pub enum LmDeployError {
  /// Invalid request parameters
  InvalidRequest(String)

  /// Model not found
  ModelNotFound(String)

  /// Tokenization error
  TokenizeError(String)

  /// TurboMind engine error
  EngineError(String)

  /// Internal server error
  InternalError(String)

  /// Timeout error
  TimeoutError(String)

  /// Rate limit exceeded
  RateLimitExceeded(String)
}

/// Convert error to HTTP status code
pub fn LmDeployError::status_code(self : LmDeployError) -> Int {
  match self {
    InvalidRequest(_) => 400
    ModelNotFound(_) => 404
    TokenizeError(_) => 400
    EngineError(_) => 500
    InternalError(_) => 500
    TimeoutError(_) => 504
    RateLimitExceeded(_) => 429
  }
}

/// Convert error to error response
pub fn LmDeployError::to_response(self : LmDeployError) -> ErrorResponse {
  let (message, type_) = match self {
    InvalidRequest(msg) => (msg, "invalid_request_error")
    ModelNotFound(msg) => (msg, "model_not_found")
    TokenizeError(msg) => (msg, "tokenize_error")
    EngineError(msg) => (msg, "engine_error")
    InternalError(msg) => (msg, "internal_error")
    TimeoutError(msg) => (msg, "timeout_error")
    RateLimitExceeded(msg) => (msg, "rate_limit_exceeded")
  }

  ErrorResponse {
    message,
    type_,
    code: self.status_code(),
    param: None,
    object: "error",
  }
}

/// Result type alias for LMDeploy operations
pub type LmResult[T] = Result[T, LmDeployError]

/// Create invalid request error
pub fn invalid_request(msg: String) -> LmDeployError {
  InvalidRequest(msg)
}

/// Create model not found error
pub fn model_not_found(model: String) -> LmDeployError {
  ModelNotFound(model)
}

/// Create tokenize error
pub fn tokenize_error(msg: String) -> LmDeployError {
  TokenizeError(msg)
}

/// Create engine error
pub fn engine_error(msg: String) -> LmDeployError {
  EngineError(msg)
}

/// Create internal error
pub fn internal_error(msg: String) -> LmDeployError {
  InternalError(msg)
}

/// Create timeout error
pub fn timeout_error(msg: String) -> LmDeployError {
  TimeoutError(msg)
}

/// Create rate limit error
pub fn rate_limit_exceeded(msg: String) -> LmDeployError {
  RateLimitExceeded(msg)
}
