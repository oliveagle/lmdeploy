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
            AppError::Other(_) => 500,
        }
    }
}
