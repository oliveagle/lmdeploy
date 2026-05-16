// Protobuf generated code is built by build.rs → src/grpc/lmdeploy.v1.rs
pub mod lmdeploy {
    pub mod v1 {
        include!(concat!(env!("OUT_DIR"), "/lmdeploy.v1.rs"));
    }
}

pub use lmdeploy::v1::lm_deploy_service_server::LmDeployServiceServer;

mod service_impl;
mod handler;

pub use handler::start_grpc_server;
pub use service_impl::LmDeployServiceImpl;
