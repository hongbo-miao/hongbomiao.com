mod config;
mod shared;

use crate::config::AppConfig;
use crate::shared::nats::services::audio_stream_processor::process_audio_stream_from_nats;
use anyhow::Result;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    tracing_subscriber::fmt()
        .with_max_level(config.log_level)
        .init();

    match process_audio_stream_from_nats().await {
        Ok(()) => {
            info!("Audio stream processing completed successfully");
            Ok(())
        }
        Err(error) => {
            error!("Failed to process audio stream: {error}");
            Err(anyhow::anyhow!("Audio stream processing failed: {error}"))
        }
    }
}
