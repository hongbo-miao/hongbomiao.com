use anyhow::Result;
use prost::Message;
use pulsar::{Producer, SerializeMessage, TokioExecutor, producer};
use tracing::info;

use crate::shared::audio::models::audio_chunk::AudioChunk;

struct AudioChunkMessage {
    payload: Vec<u8>,
    partition_key: String,
}

impl SerializeMessage for AudioChunkMessage {
    fn serialize_message(input: Self) -> Result<producer::Message, pulsar::Error> {
        Ok(producer::Message {
            payload: input.payload,
            partition_key: Some(input.partition_key),
            ..Default::default()
        })
    }
}

pub async fn publish_audio_chunk(
    audio_chunk_producer: &mut Producer<TokioExecutor>,
    audio_chunk: AudioChunk,
    published_chunk_count: &mut u64,
) -> Result<()> {
    let device_id = audio_chunk.device_id.clone();
    let payload = audio_chunk.encode_to_vec();

    audio_chunk_producer
        .send_non_blocking(AudioChunkMessage {
            payload,
            partition_key: device_id.clone(),
        })
        .await
        .map_err(|error| anyhow::anyhow!("Failed to send audio chunk to Pulsar: {error}"))?
        .await
        .map_err(|error| {
            anyhow::anyhow!("Failed to receive Pulsar send acknowledgement: {error}")
        })?;

    *published_chunk_count += 1;

    if published_chunk_count.is_multiple_of(50) {
        info!("Device {device_id}: published {published_chunk_count} audio chunks to Pulsar");
    }

    Ok(())
}
