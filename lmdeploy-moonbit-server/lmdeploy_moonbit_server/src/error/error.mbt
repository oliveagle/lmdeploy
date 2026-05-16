// src/error/error.mbt
// Error types and handling for LMDeploy MoonBit server
// Reference: lmdeploy-rust-server/src/error.rs, Python api_server.py ErrorResponse

/// Error response structure matching OpenAI API error format
pub struct ErrorResponse {
  message: String
  type_: String
  code: Int
  param: Option[String]
  object: String
}

pub fn ErrorResponse::default() -> ErrorResponse {
  { message: "", type_: "error", code: 500, param: None, object: "error" }
}

pub fn ErrorResponse::new(message: String, type_: String, code: Int) -> ErrorResponse {
  { message, type_, code, param: None, object: "error" }
}

pub fn ErrorResponse::to_json(self: ErrorResponse) -> String {
  """
{"error": {"message": """" ++ self.message ++ """", "type": """" ++ self.type_ ++ """", "code": """ ++ int_to_string(self.code) ++ """}}
"""
}

/// Build error JSON string
pub fn build_error_json(code: Int, type_: String, message: String) -> String {
  """
{"error": {"message": """" ++ message ++ """", "type": """" ++ type_ ++ """", "code": """ ++ int_to_string(code) ++ """}}
"""
}

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

pub fn LmDeployError::status_code(self: LmDeployError) -> Int {
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

pub fn LmDeployError::to_response(self: LmDeployError) -> ErrorResponse {
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

pub fn LmDeployError::to_json(self: LmDeployError) -> String {
  let resp = self.to_response()
  resp.to_json()
}

pub fn LmDeployError::to_string(self: LmDeployError) -> String {
  let resp = self.to_response()
  resp.message
}

pub type LmResult[T] = Result[T, LmDeployError]

pub fn invalid_request(msg: String) -> LmDeployError {
  InvalidRequest(msg)
}

pub fn model_not_found(model: String) -> LmDeployError {
  ModelNotFound(model)
}

pub fn tokenize_error(msg: String) -> LmDeployError {
  TokenizeError(msg)
}

pub fn engine_error(msg: String) -> LmDeployError {
  EngineError(msg)
}

pub fn internal_error(msg: String) -> LmDeployError {
  InternalError(msg)
}

pub fn timeout_error(msg: String) -> LmDeployError {
  TimeoutError(msg)
}

pub fn rate_limit_exceeded(msg: String) -> LmDeployError {
  RateLimitExceeded(msg)
}