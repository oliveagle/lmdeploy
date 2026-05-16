use axum::{
    extract::Request,
    http::StatusCode,
    response::{Json, Response},
};
use std::time::Duration;
use tower::{Layer, Service};

use crate::error::{AppError, ErrorResponse};

/// Error response wrapper for Axum handlers
pub fn error_response(err: &AppError) -> (StatusCode, Json<ErrorResponse>) {
    let status = StatusCode::from_u16(err.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let error_resp = ErrorResponse::from(err);
    tracing::error!(
        status = status.as_u16(),
        error_type = error_resp.error_type,
        message = %error_resp.message,
        "Request error"
    );
    (status, Json(error_resp))
}

/// Middleware for request timeout
#[derive(Clone)]
pub struct TimeoutLayer {
    duration: Duration,
}

impl TimeoutLayer {
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
}

impl<S> Layer<S> for TimeoutLayer {
    type Service = TimeoutService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TimeoutService {
            inner,
            duration: self.duration,
        }
    }
}

#[derive(Clone)]
pub struct TimeoutService<S> {
    inner: S,
    duration: Duration,
}

impl<S> Service<Request> for TimeoutService<S>
where
    S: Service<Request, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Into<AppError> + Send,
{
    type Response = S::Response;
    type Error = AppError;
    type Future = futures::future::BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let duration = self.duration;
        let mut inner = self.inner.clone();

        Box::pin(async move {
            match tokio::time::timeout(duration, inner.call(req)).await {
                Ok(Ok(resp)) => Ok(resp),
                Ok(Err(e)) => Err(e.into()),
                Err(_) => {
                    tracing::warn!(
                        duration_ms = duration.as_millis(),
                        "Request timeout"
                    );
                    Err(AppError::RequestTimeout)
                }
            }
        })
    }
}
