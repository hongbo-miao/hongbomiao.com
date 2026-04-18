#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod shared;

use anyhow::Result;
use tracing::info;

use crate::config::AppConfig;
use crate::shared::audio::services::stream_audio_to_ingest::stream_audio_to_ingest;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::get();

    info!(
        "Device {}: connecting to ingest at {}",
        config.device_id, config.ingest_url
    );

    stream_audio_to_ingest(&config.device_id, &config.ingest_url, &config.wav_path).await
}
