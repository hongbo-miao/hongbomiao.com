#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod shared;

use std::net::SocketAddr;
use std::sync::Arc;

use crate::config::AppConfig;
use crate::shared::audio::services::handle_audio_stream::{
    AudioIngestServiceImpl, audio_ingest::audio_ingest_service_server::AudioIngestServiceServer,
};
use anyhow::{Result, anyhow};
use pulsar::{Pulsar, TokioExecutor};
use tokio::sync::Mutex;
use tonic::transport::Server;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::get();

    info!("Connecting to Pulsar at {}", config.pulsar_url);
    let pulsar_client: Pulsar<TokioExecutor> = Pulsar::builder(&config.pulsar_url, TokioExecutor)
        .build()
        .await?;

    let pulsar_producer = Arc::new(Mutex::new(
        pulsar_client
            .producer()
            .with_topic(&config.pulsar_topic)
            .build()
            .await?,
    ));

    let service = AudioIngestServiceImpl { pulsar_producer };
    let addr = SocketAddr::from(([0, 0, 0, 0], config.grpc_port));

    info!("gRPC server listening on {addr}");

    Server::builder()
        .add_service(AudioIngestServiceServer::new(service))
        .serve(addr)
        .await
        .map_err(|error| anyhow!("gRPC server error: {error}"))?;

    Ok(())
}
