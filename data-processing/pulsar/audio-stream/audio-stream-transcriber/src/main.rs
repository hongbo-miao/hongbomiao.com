#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod shared;

use anyhow::{Result, anyhow};
use tracing::info;

use crate::config::AppConfig;
use crate::shared::audio::services::transcribe_audio_stream::transcribe_audio_stream;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("warn,audio_stream_transcriber=info")
            }),
        )
        .init();

    let app_config = AppConfig::load()?;

    info!(
        "Starting audio-stream-transcriber as {}",
        app_config.instance_id
    );

    transcribe_audio_stream(
        &app_config.pulsar_url,
        &app_config.pulsar_topic,
        &app_config.pulsar_output_topic,
        &app_config.livekit_url,
        &app_config.livekit_api_key,
        &app_config.livekit_api_secret,
        &app_config.livekit_room,
        &app_config.instance_id,
        &app_config.asr_model_dir,
        &app_config.silero_vad_model_dir,
    )
    .await
    .map_err(|error| anyhow!("audio-stream-transcriber failed: {error}"))
}
