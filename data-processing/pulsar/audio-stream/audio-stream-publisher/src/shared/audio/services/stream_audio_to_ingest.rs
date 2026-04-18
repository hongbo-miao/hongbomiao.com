use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use async_stream::stream;
use hound::WavReader;
use tonic::transport::Channel;
use tracing::info;

pub mod audio_ingest {
    tonic::include_proto!("audio_ingest");
}

use audio_ingest::AudioFrame;
use audio_ingest::audio_ingest_service_client::AudioIngestServiceClient;

const OUTPUT_SAMPLE_RATE_HZ: u32 = 16_000;
const CHANNEL_COUNT: u32 = 1;
const FRAME_DURATION_MS: u64 = 20;
const FRAME_SAMPLE_COUNT: usize =
    (OUTPUT_SAMPLE_RATE_HZ as usize * FRAME_DURATION_MS as usize) / 1_000;

pub async fn stream_audio_to_ingest(
    device_id: &str,
    ingest_url: &str,
    wav_path: &str,
) -> Result<()> {
    let samples = load_wav_as_16khz_mono(wav_path)?;
    info!(
        "Device {device_id}: loaded WAV with {} samples at 16kHz",
        samples.len()
    );

    let channel = Channel::from_shared(ingest_url.to_string())
        .map_err(|error| anyhow!("Invalid ingest URL: {error}"))?
        .connect()
        .await
        .map_err(|error| anyhow!("Failed to connect to ingest service: {error}"))?;

    let mut client = AudioIngestServiceClient::new(channel);

    info!("Device {device_id}: connected to ingest, starting audio stream");

    let device_id_owned = device_id.to_string();
    let outbound = stream! {
        let mut sample_offset: usize = 0;
        let mut frame_index: u64 = 0;
        let start_instant = tokio::time::Instant::now();

        loop {
            let timestamp_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("System clock error")
                .as_nanos() as i64;

            let end = sample_offset + FRAME_SAMPLE_COUNT;
            let frame_samples: Vec<i16> = if end <= samples.len() {
                samples[sample_offset..end].to_vec()
            } else {
                let mut buf = samples[sample_offset..].to_vec();
                let remaining = FRAME_SAMPLE_COUNT - buf.len();
                buf.extend_from_slice(&samples[..remaining]);
                buf
            };
            sample_offset = (sample_offset + FRAME_SAMPLE_COUNT) % samples.len();

            let pcm_data = pcm_samples_to_bytes(&frame_samples);

            yield AudioFrame {
                device_id: device_id_owned.clone(),
                pcm_data,
                timestamp_ns,
                sample_rate_hz: OUTPUT_SAMPLE_RATE_HZ,
                channel_count: CHANNEL_COUNT,
            };

            frame_index += 1;
            if frame_index.is_multiple_of(50) {
                info!("Device {device_id_owned}: sent frame {frame_index}");
            }

            let next_deadline = start_instant + Duration::from_millis(frame_index * FRAME_DURATION_MS);
            tokio::time::sleep_until(next_deadline).await;
        }
    };

    client
        .stream_audio(tonic::Request::new(outbound))
        .await
        .map_err(|error| anyhow!("gRPC stream error: {error}"))?
        .into_inner();

    Ok(())
}

fn load_wav_as_16khz_mono(wav_path: &str) -> Result<Vec<i16>> {
    let mut reader = WavReader::open(wav_path)
        .map_err(|error| anyhow!("Failed to open WAV file {wav_path}: {error}"))?;

    let spec = reader.spec();
    info!(
        "WAV spec: {}Hz {} channels {:?}",
        spec.sample_rate, spec.channels, spec.sample_format
    );

    let raw_samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| anyhow!("Failed to read WAV samples: {error}"))?,
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| anyhow!("Failed to read WAV samples: {error}"))?
            .into_iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect(),
    };

    let mono_samples: Vec<i16> = if spec.channels == 2 {
        raw_samples
            .chunks_exact(2)
            .map(|pair| ((pair[0] as i32 + pair[1] as i32) / 2) as i16)
            .collect()
    } else {
        raw_samples
    };

    if spec.sample_rate == OUTPUT_SAMPLE_RATE_HZ {
        return Ok(mono_samples);
    }

    let ratio = spec.sample_rate as f64 / OUTPUT_SAMPLE_RATE_HZ as f64;
    let output_len = (mono_samples.len() as f64 / ratio) as usize;
    let resampled = (0..output_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f64;
            let s0 = mono_samples[src_idx.min(mono_samples.len() - 1)] as f64;
            let s1 = mono_samples[(src_idx + 1).min(mono_samples.len() - 1)] as f64;
            (s0 + frac * (s1 - s0)) as i16
        })
        .collect();

    Ok(resampled)
}

fn pcm_samples_to_bytes(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&sample| sample.to_le_bytes())
        .collect()
}
