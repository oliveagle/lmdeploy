use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::Server;

use crate::cache::TokenizeCache;

use super::service_impl::LmDeployServiceImpl;
use super::LmDeployServiceServer;

pub async fn start_grpc_server(
    addr: SocketAddr,
    version: String,
    tokenizer_cache: Arc<TokenizeCache>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = LmDeployServiceImpl {
        version,
        tokenizer_cache,
    };

    tracing::info!(%addr, "Starting gRPC server");

    Server::builder()
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .http2_keepalive_timeout(Some(Duration::from_secs(10)))
        .add_service(LmDeployServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
