mod config;
mod shared;

mod audio_chunk_capnp {
    include!(concat!(env!("OUT_DIR"), "/audio_chunk_capnp.rs"));
}

mod transcription_capnp {
    include!(concat!(env!("OUT_DIR"), "/transcription_capnp.rs"));
}

use anyhow::Result;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::config::AppConfig;
use crate::shared::nats::services::process_audio_stream_from_nats::process_audio_stream_from_nats;

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    let env_filter = EnvFilter::new(format!("{},flacenc=warn", config.log_level));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

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
