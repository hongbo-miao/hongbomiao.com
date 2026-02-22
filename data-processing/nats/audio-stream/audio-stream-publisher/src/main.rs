mod config;

mod audio_chunk_capnp {
    include!(concat!(env!("OUT_DIR"), "/audio_chunk_capnp.rs"));
}

use anyhow::{Result, anyhow};
use async_nats::jetstream::context::PublishAckFuture;
use ffmpeg_sidecar::child::FfmpegChild;
use ffmpeg_sidecar::command::FfmpegCommand;
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::Verify;
use tokio::io::AsyncReadExt;
use tokio::process::ChildStdout;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::config::AppConfig;

fn encode_pcm_i16_to_flac_bytes(
    pcm_samples: &[i16],
    sample_rate_hz: u32,
    channel_count: u32,
) -> Result<Vec<u8>> {
    let samples_i32: Vec<i32> = pcm_samples.iter().map(|&sample| sample as i32).collect();

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, verify_error)| anyhow!("Config verification failed: {verify_error:?}"))?;

    let source = flacenc::source::MemSource::from_samples(
        &samples_i32,
        channel_count as usize,
        16,
        sample_rate_hz as usize,
    );

    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|error| anyhow!("Encode failed: {error}"))?;

    let mut sink = ByteSink::new();
    flac_stream
        .write(&mut sink)
        .map_err(|error| anyhow!("Write FLAC stream failed: {error}"))?;

    Ok(sink.as_slice().to_vec())
}

fn spawn_ffmpeg_process(
    stream_url: &str,
    sample_rate_hz: u32,
) -> Result<FfmpegChild, std::io::Error> {
    FfmpegCommand::new()
        .input(stream_url)
        .arg("-nostdin")
        .hide_banner()
        .args(["-loglevel", "quiet"])
        .args(["-ac", "1", "-ar", &sample_rate_hz.to_string()])
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
        if buffer.len() >= chunk_size {
            let chunk = buffer[..chunk_size].to_vec();
            buffer.drain(..chunk_size);
            return Ok(Some(chunk));
        }

        let bytes_read = reader.read(&mut temp_buffer).await?;

        if bytes_read == 0 {
            return Ok(None);
        }

        buffer.extend_from_slice(&temp_buffer[..bytes_read]);
    }
}

fn serialize_audio_chunk(
    timestamp_ns: i64,
    sample_rate_hz: u32,
    audio_data: &[u8],
    audio_format: &str,
) -> Result<Vec<u8>> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut audio_chunk = message.init_root::<audio_chunk_capnp::audio_chunk::Builder>();
        audio_chunk.set_timestamp_ns(timestamp_ns);
        audio_chunk.set_sample_rate_hz(sample_rate_hz);
        audio_chunk.set_audio_data(audio_data);
        audio_chunk.set_audio_format(audio_format);
    }
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message)?;
    Ok(bytes)
}

async fn publish_audio_stream(
    config: &AppConfig,
    jetstream_context: async_nats::jetstream::Context,
    stream_url: &str,
    subject: &str,
) -> anyhow::Result<usize> {
    let mut published_chunk_count: usize = 0;
    let mut ffmpeg_process = spawn_ffmpeg_process(stream_url, config.sample_rate_hz)?;

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
            Ok(Some(pcm_chunk)) => {
                published_chunk_count += 1;

                let pcm_samples: Vec<i16> = pcm_chunk
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect();

                let flac_data =
                    encode_pcm_i16_to_flac_bytes(&pcm_samples, config.sample_rate_hz, 1)?;

                let timestamp_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                let message_bytes =
                    serialize_audio_chunk(timestamp_ns, config.sample_rate_hz, &flac_data, "flac")?;

                let mut headers = async_nats::HeaderMap::new();
                headers.insert("Nats-Msg-Id", Uuid::new_v4().to_string());

                let publish_acknowledgement: PublishAckFuture = jetstream_context
                    .publish_with_headers(subject.to_string(), headers, message_bytes.into())
                    .await?;

                if published_chunk_count.is_multiple_of(50) {
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

    if let Err(error) = ffmpeg_process.kill() {
        error!("Failed to kill FFmpeg process: {error}");
    }

    Ok(published_chunk_count)
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    let env_filter = EnvFilter::new(format!("{},flacenc=warn", config.log_level));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();
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
