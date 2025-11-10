mod config;

use crate::config::AppConfig;
use anyhow::Result;
use async_nats::jetstream::context::PublishAckFuture;
use ffmpeg_sidecar::child::FfmpegChild;
use ffmpeg_sidecar::command::FfmpegCommand;
use tokio::io::AsyncReadExt;
use tokio::process::ChildStdout;
use tracing::{error, info};
use uuid::Uuid;

fn spawn_ffmpeg_process(stream_url: &str) -> Result<FfmpegChild, std::io::Error> {
    FfmpegCommand::new()
        .input(stream_url)
        .arg("-nostdin")
        .hide_banner()
        .args(["-loglevel", "quiet"])
        .filter_complex(
            "pan=mono|c0=0.707107*FL+0.707107*FR,aresample=resampler=soxr:sample_rate=16000",
        )
        .format("s16le")
        .pipe_stdout()
        .spawn()
}

async fn read_pcm_chunk(
    reader: &mut ChildStdout,
    buffer: &mut Vec<u8>,
    pcm_chunk_size_bytes: u32,
) -> Result<Option<Vec<u8>>, std::io::Error> {
    let chunk_size = pcm_chunk_size_bytes as usize;
    let mut temp_buffer = vec![0u8; chunk_size * 2];

    loop {
        // If we have enough data for a chunk, return it
        if buffer.len() >= chunk_size {
            // Prepend 4-byte length prefix (little-endian)
            let mut message = Vec::with_capacity(4 + chunk_size);
            message.extend_from_slice(&(chunk_size as u32).to_le_bytes());
            message.extend_from_slice(&buffer[..chunk_size]);
            buffer.drain(..chunk_size);
            return Ok(Some(message));
        }

        // Read more data into buffer
        let bytes_read = reader.read(&mut temp_buffer).await?;

        if bytes_read == 0 {
            // No more data available
            return Ok(None);
        }

        buffer.extend_from_slice(&temp_buffer[..bytes_read]);
    }
}

async fn publish_audio_stream(
    config: &AppConfig,
    jetstream_context: async_nats::jetstream::Context,
    stream_url: &str,
    subject: &str,
) -> anyhow::Result<usize> {
    let mut published_chunk_count: usize = 0;
    let mut ffmpeg_process = spawn_ffmpeg_process(stream_url)?;

    info!(
        "Streaming '{}' from {}",
        config.emergency_stream_name, config.emergency_stream_location
    );

    let stdout = ffmpeg_process
        .as_inner_mut()
        .stdout
        .take()
        .expect("Failed to get FFmpeg stdout");

    let mut reader = ChildStdout::from_std(stdout).expect("Failed to convert to tokio ChildStdout");
    let mut buffer: Vec<u8> = Vec::new();

    loop {
        match read_pcm_chunk(&mut reader, &mut buffer, config.pcm_chunk_size_bytes).await {
            Ok(Some(message_to_publish)) => {
                published_chunk_count += 1;

                let mut headers = async_nats::HeaderMap::new();
                headers.insert("Nats-Msg-Id", Uuid::new_v4().to_string());

                let publish_acknowledgement: PublishAckFuture = jetstream_context
                    .publish_with_headers(subject.to_string(), headers, message_to_publish.into())
                    .await?;

                if published_chunk_count % 50 == 0 {
                    let ack = publish_acknowledgement.await?;
                    info!(
                        "Published {published_chunk_count} audio chunks (latest seq: {})",
                        ack.sequence
                    );
                }
            }
            Ok(None) => {
                break;
            }
            Err(error) => {
                error!("Failed to read PCM chunk: {error}");
                break;
            }
        }
    }

    // Terminate FFmpeg process
    if let Err(error) = ffmpeg_process.kill() {
        error!("Failed to kill FFmpeg process: {error}");
    }

    Ok(published_chunk_count)
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    tracing_subscriber::fmt()
        .with_max_level(config.log_level)
        .init();
    ffmpeg_sidecar::download::auto_download()?;

    info!("Connecting to NATS at {}", config.nats_url);

    let nats_client = async_nats::connect(&config.nats_url).await?;
    let jetstream_context = async_nats::jetstream::new(nats_client.clone());

    let subject = format!(
        "{}.{}",
        config.subject_prefix, config.emergency_stream_identifier
    );

    match publish_audio_stream(
        config,
        jetstream_context,
        &config.emergency_stream_url,
        &subject,
    )
    .await
    {
        Ok(published_chunk_count) => {
            info!("Audio streaming completed with {published_chunk_count} chunks published");
        }
        Err(error) => {
            error!("Failed to publish audio stream: {error}");
            return Err(error);
        }
    }

    nats_client.drain().await?;
    info!("NATS connection closed");

    Ok(())
}
