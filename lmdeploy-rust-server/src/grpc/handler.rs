use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::Server;

use crate::cache::TokenizeCache;
use crate::model::ModelManager;

use super::service_impl::LmDeployServiceImpl;
use super::LmDeployServiceServer;

pub async fn start_grpc_server(
    addr: SocketAddr,
    version: String,
    tokenizer_cache: Arc<TokenizeCache>,
    model_manager: Arc<RwLock<ModelManager>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = LmDeployServiceImpl::new(version, tokenizer_cache, model_manager);

    tracing::info!(%addr, "Starting gRPC server");

    Server::builder()
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .http2_keepalive_timeout(Some(Duration::from_secs(10)))
        .add_service(LmDeployServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
