use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("TurboMind error: {0}")]
    TurboMind(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already loaded: {0}")]
    ModelAlreadyLoaded(String),

    #[error("Cannot unload the default model")]
    CannotUnloadDefaultModel,

    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Request timeout")]
    RequestTimeout,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AppError>;

impl AppError {
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::InvalidRequest(_) => 400,
            AppError::TurboMind(_) => 500,
            AppError::Internal(_) => 500,
            AppError::Config(_) => 500,
            AppError::Io(_) => 500,
            AppError::Serde(_) => 400,
            AppError::ModelNotFound(_) => 404,
            AppError::ModelAlreadyLoaded(_) => 409,
            AppError::CannotUnloadDefaultModel => 400,
            AppError::ModelLoadFailed(_) => 500,
            AppError::InferenceFailed(_) => 500,
            AppError::RateLimitExceeded(_) => 429,
            AppError::RequestTimeout => 408,
            AppError::ServiceUnavailable(_) => 503,
            AppError::Other(_) => 500,
        }
    }

    pub fn error_type(&self) -> &'static str {
        match self {
            AppError::InvalidRequest(_) => "invalid_request_error",
            AppError::RateLimitExceeded(_) => "rate_limit_exceeded",
            AppError::RequestTimeout => "request_timeout",
            AppError::ServiceUnavailable(_) => "service_unavailable",
            AppError::TurboMind(_) => "turbomind_error",
            AppError::ModelNotFound(_) => "model_not_found",
            AppError::ModelLoadFailed(_) => "model_load_failed",
            _ => "internal_error",
        }
    }
}

/// OpenAI-compatible error response format
#[derive(serde::Serialize)]
pub struct ErrorResponse {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: u16,
}

impl From<&AppError> for ErrorResponse {
    fn from(err: &AppError) -> Self {
        Self {
            message: err.to_string(),
            error_type: err.error_type().to_string(),
            code: err.status_code(),
        }
    }
}
