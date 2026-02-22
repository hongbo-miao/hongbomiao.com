mod config;
mod shared;

mod transcription_capnp {
    include!(concat!(env!("OUT_DIR"), "/transcription_capnp.rs"));
}

use anyhow::Result;
use tracing::{error, info};

use crate::config::AppConfig;
use crate::shared::nats::services::process_transcription_stream_from_nats::process_transcription_stream_from_nats;

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    tracing_subscriber::fmt()
        .with_max_level(config.log_level)
        .init();

    match process_transcription_stream_from_nats().await {
        Ok(()) => {
            info!("Transcription stream sinking completed successfully");
            Ok(())
        }
        Err(error) => {
            error!("Failed to process transcription stream: {error}");
            Err(anyhow::anyhow!(
                "Transcription stream processing failed: {error}"
            ))
        }
    }
}
