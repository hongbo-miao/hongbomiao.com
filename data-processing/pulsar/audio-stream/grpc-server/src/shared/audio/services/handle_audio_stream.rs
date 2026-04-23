use std::sync::Arc;

use pulsar::{Producer, TokioExecutor};
use tokio::sync::Mutex;
use tonic::{Request, Response, Status, Streaming};
use tracing::{error, info};

use crate::shared::audio::services::publish_audio_chunk::publish_audio_chunk;
use crate::shared::audio::types::audio_chunk::AudioChunk;

pub mod audio_ingest {
    tonic::include_proto!("audio_ingest");
}

use audio_ingest::audio_ingest_service_server::AudioIngestService;
use audio_ingest::{AudioFrame, StreamAck};

const CHUNK_DURATION_MS: i32 = 20;
const OPUS_FRAME_DURATION_MS: usize = 20;
const OPUS_MAX_PACKET_SIZE: usize = 4_000;

pub struct AudioIngestServiceImpl {
    pub pulsar_producer: Arc<Mutex<Producer<TokioExecutor>>>,
}

#[tonic::async_trait]
impl AudioIngestService for AudioIngestServiceImpl {
    async fn stream_audio(
        &self,
        request: Request<Streaming<AudioFrame>>,
    ) -> Result<Response<StreamAck>, Status> {
        let mut stream = request.into_inner();
        let pulsar_producer = Arc::clone(&self.pulsar_producer);

        let mut pcm_buffer: Vec<i16> = Vec::new();
        let mut opus_encoder: Option<opus::Encoder> = None;
        let mut device_id: Option<String> = None;
        let mut sample_rate_hz: u32 = 16_000;
        let mut frames_received: u64 = 0;
        let mut published_chunk_count: u64 = 0;
        let mut sequence_number: u64 = 0;

        while let Some(frame_result) = stream.message().await? {
            frames_received += 1;

            if device_id.is_none() {
                device_id = Some(frame_result.device_id.clone());
                sample_rate_hz = frame_result.sample_rate_hz;

                let encoder = opus::Encoder::new(
                    sample_rate_hz,
                    opus::Channels::Mono,
                    opus::Application::Audio,
                )
                .map_err(|error| {
                    Status::internal(format!("Failed to create Opus encoder: {error}"))
                })?;
                opus_encoder = Some(encoder);

                info!(
                    "Device {}: ingest stream started at {}Hz",
                    frame_result.device_id, sample_rate_hz
                );
            }

            let pcm_samples = bytes_to_pcm_samples(&frame_result.pcm_data);
            pcm_buffer.extend_from_slice(&pcm_samples);

            let chunk_sample_count = (sample_rate_hz as usize * CHUNK_DURATION_MS as usize) / 1_000;
            let opus_frame_sample_count =
                (sample_rate_hz as usize * OPUS_FRAME_DURATION_MS) / 1_000;
            let opus_frames_per_chunk = CHUNK_DURATION_MS as usize / OPUS_FRAME_DURATION_MS;

            while pcm_buffer.len() >= chunk_sample_count {
                let chunk_samples: Vec<i16> = pcm_buffer.drain(..chunk_sample_count).collect();

                let encoder = opus_encoder.as_mut().expect("Encoder initialized above");
                let mut opus_packets: Vec<Vec<u8>> = Vec::with_capacity(opus_frames_per_chunk);
                for frame_samples in chunk_samples.chunks(opus_frame_sample_count) {
                    let packet = encoder
                        .encode_vec(frame_samples, OPUS_MAX_PACKET_SIZE)
                        .map_err(|error| {
                            Status::internal(format!("Failed to encode Opus frame: {error}"))
                        })?;
                    opus_packets.push(packet);
                }

                let audio_chunk = AudioChunk {
                    device_id: device_id.as_deref().unwrap_or("unknown").to_string(),
                    timestamp_ns: frame_result.timestamp_ns,
                    opus_packets,
                    duration_ms: CHUNK_DURATION_MS,
                    sample_rate_hz: sample_rate_hz as i32,
                    channel_count: 1,
                    sequence_number,
                };

                sequence_number += 1;

                let mut producer = pulsar_producer.lock().await;
                publish_audio_chunk(&mut producer, audio_chunk, &mut published_chunk_count)
                    .await
                    .map_err(|error| {
                        error!("Failed to publish audio chunk: {error}");
                        Status::internal(format!("Failed to publish to Pulsar: {error}"))
                    })?;
            }
        }

        info!(
            "Device {}: ingest stream ended after {frames_received} frames",
            device_id.as_deref().unwrap_or("unknown")
        );

        Ok(Response::new(StreamAck { frames_received }))
    }
}

fn bytes_to_pcm_samples(data: &[u8]) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}
