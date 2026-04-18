use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use futures::StreamExt;
use livekit::options::TrackPublishOptions;
use livekit::prelude::*;
use livekit::webrtc::audio_source::native::NativeAudioSource;
use livekit::webrtc::prelude::{AudioFrame, AudioSourceOptions, RtcAudioSource};
use prost::Message;
use pulsar::consumer::InitialPosition;
use pulsar::{
    Consumer, ConsumerOptions, DeserializeMessage, Payload, Pulsar, SubType, TokioExecutor,
};
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::shared::audio::models::audio_chunk::AudioChunk;

const PCM_SAMPLE_RATE_HZ: u32 = 16_000;
const STALE_CHUNK_THRESHOLD_MS: u128 = 2_000;
const OPUS_FRAME_SAMPLE_COUNT: usize = 320; // 20ms @ 16kHz
const DEVICE_CHANNEL_BUFFER: usize = 16;
const DEVICE_STALE_TIMEOUT_SECS: u64 = 5;

struct RawPayload;

impl DeserializeMessage for RawPayload {
    type Output = Vec<u8>;

    fn deserialize_message(payload: &Payload) -> Self::Output {
        payload.data.clone()
    }
}

pub async fn bridge_pulsar_to_livekit(
    pulsar_url: &str,
    pulsar_topic: &str,
    livekit_url: &str,
    livekit_api_key: &str,
    livekit_api_secret: &str,
    livekit_room: &str,
    bridge_instance_id: &str,
) -> Result<()> {
    info!("Connecting to Pulsar at {pulsar_url}");
    let pulsar_client: Pulsar<TokioExecutor> =
        Pulsar::builder(pulsar_url, TokioExecutor).build().await?;

    let subscription_name = "pulsar-livekit-bridge-subscription";
    let mut consumer: Consumer<RawPayload, TokioExecutor> = pulsar_client
        .consumer()
        .with_topic(pulsar_topic)
        .with_subscription(subscription_name)
        .with_subscription_type(SubType::KeyShared)
        .with_consumer_name(bridge_instance_id)
        .with_options(ConsumerOptions {
            initial_position: InitialPosition::Latest,
            ..Default::default()
        })
        .build()
        .await?;

    let token = generate_livekit_publish_token(
        livekit_api_key,
        livekit_api_secret,
        livekit_room,
        bridge_instance_id,
    )?;

    info!("Connecting to LiveKit at {livekit_url}, room {livekit_room}");
    let (room, _room_event_receiver) = Room::connect(livekit_url, &token, RoomOptions::default())
        .await
        .map_err(|error| anyhow!("Failed to connect to LiveKit room: {error}"))?;
    let room = Arc::new(room);

    info!("Connected to LiveKit room {livekit_room} as {bridge_instance_id}");

    let mut device_senders: HashMap<String, mpsc::Sender<AudioChunk>> = HashMap::new();
    let mut expected_sequence_number_by_device_id: HashMap<String, u64> = HashMap::new();
    let mut total_received_count: u64 = 0;
    let mut total_gap_count: u64 = 0;

    while let Some(message_result) = consumer.next().await {
        let pulsar_message = match message_result {
            Ok(message) => message,
            Err(error) => {
                error!("Failed to receive Pulsar message: {error}");
                continue;
            }
        };

        let audio_chunk = match AudioChunk::decode(pulsar_message.payload.data.as_slice()) {
            Ok(chunk) => chunk,
            Err(error) => {
                error!("Failed to decode AudioChunk: {error}");
                consumer.ack(&pulsar_message).await.ok();
                continue;
            }
        };

        // Pulsar resumes from the subscription cursor on restart, ignoring InitialPosition::Latest
        // for existing subscriptions. Skip chunks older than the threshold to avoid replaying
        // buffered messages with stale timestamps, which would inflate the latency display.
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let age_ms = now_ns.saturating_sub(audio_chunk.timestamp_ns as u128) / 1_000_000;
        if age_ms > STALE_CHUNK_THRESHOLD_MS {
            consumer.ack(&pulsar_message).await.ok();
            continue;
        }

        total_received_count += 1;
        let device_id = &audio_chunk.device_id;
        let sequence_number = audio_chunk.sequence_number;

        let expected = expected_sequence_number_by_device_id
            .entry(device_id.clone())
            .or_insert(sequence_number);
        if sequence_number != *expected {
            let gap = sequence_number as i64 - *expected as i64;
            total_gap_count += 1;
            warn!(
                "Sequence gap for device {device_id}: expected {expected}, got {sequence_number} (gap: {gap}, total_gaps: {total_gap_count})"
            );
        }
        *expected_sequence_number_by_device_id
            .entry(device_id.clone())
            .or_insert(0) = sequence_number + 1;

        if total_received_count.is_multiple_of(50) {
            info!(
                "device={device_id} sequence={sequence_number} received={total_received_count} gaps={total_gap_count}"
            );
        }

        if !device_senders.contains_key(device_id) {
            let source = Arc::new(NativeAudioSource::new(
                AudioSourceOptions::default(),
                PCM_SAMPLE_RATE_HZ,
                1,
                2_000,
            ));

            let track = LocalAudioTrack::create_audio_track(
                &format!("audio-{device_id}"),
                RtcAudioSource::Native((*source).clone()),
            );

            let publish_options = TrackPublishOptions {
                source: TrackSource::Microphone,
                red: true,
                dtx: false,
                ..Default::default()
            };

            let publication = room
                .local_participant()
                .publish_track(LocalTrack::Audio(track), publish_options)
                .await
                .map_err(|error| anyhow!("Failed to publish track for {device_id}: {error}"))?;
            let track_sid = publication.sid();

            info!("Published LiveKit audio track for device {device_id}");

            let (sender, receiver) = mpsc::channel::<AudioChunk>(DEVICE_CHANNEL_BUFFER);
            device_senders.insert(device_id.clone(), sender);

            tokio::spawn(run_device_audio_task(
                device_id.clone(),
                source,
                receiver,
                room.clone(),
                track_sid,
            ));
        }

        let device_id_owned = device_id.clone();
        let send_failed = match device_senders.get(&device_id_owned) {
            Some(sender) => sender.send(audio_chunk).await.is_err(),
            None => false,
        };
        if send_failed {
            device_senders.remove(&device_id_owned);
        }

        consumer
            .ack(&pulsar_message)
            .await
            .map_err(|error| anyhow!("Failed to acknowledge Pulsar message: {error}"))?;
    }

    Ok(())
}

async fn run_device_audio_task(
    device_id: String,
    audio_source: Arc<NativeAudioSource>,
    mut chunk_receiver: mpsc::Receiver<AudioChunk>,
    room: Arc<Room>,
    track_sid: TrackSid,
) {
    let mut decoder = match opus::Decoder::new(PCM_SAMPLE_RATE_HZ, opus::Channels::Mono) {
        Ok(decoder) => decoder,
        Err(error) => {
            error!("Failed to create Opus decoder for {device_id}: {error}");
            return;
        }
    };

    loop {
        let audio_chunk = match tokio::time::timeout(
            Duration::from_secs(DEVICE_STALE_TIMEOUT_SECS),
            chunk_receiver.recv(),
        )
        .await
        {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(_) => {
                info!(
                    "Device {device_id}: no audio for {DEVICE_STALE_TIMEOUT_SECS}s, unpublishing track"
                );
                break;
            }
        };

        let sequence_number = audio_chunk.sequence_number;
        let opus_packet_count = audio_chunk.opus_packets.len() as u64;

        let mut pcm_buffer = vec![0i16; OPUS_FRAME_SAMPLE_COUNT];
        for opus_packet in &audio_chunk.opus_packets {
            let decoded_sample_count = match decoder.decode(opus_packet, &mut pcm_buffer, false) {
                Ok(count) => count,
                Err(error) => {
                    error!("Failed to decode Opus packet for {device_id}: {error}");
                    continue;
                }
            };

            let audio_frame = AudioFrame {
                data: pcm_buffer[..decoded_sample_count].into(),
                samples_per_channel: decoded_sample_count as u32,
                sample_rate: PCM_SAMPLE_RATE_HZ,
                num_channels: 1,
            };

            if let Err(error) = audio_source.capture_frame(&audio_frame).await {
                error!("Failed to capture audio frame for {device_id}: {error}");
            }
        }

        let frame_count = (sequence_number + 1) * opus_packet_count;
        let seq_payload = json!({
            "device_id": &device_id,
            "seq": frame_count,
            "timestamp_ns": audio_chunk.timestamp_ns
        })
        .to_string()
        .into_bytes();

        room.local_participant()
            .publish_data(DataPacket {
                payload: seq_payload,
                topic: Some("audio-seq".to_string()),
                reliable: false,
                destination_identities: vec![],
            })
            .await
            .ok();
    }

    if let Err(error) = room.local_participant().unpublish_track(&track_sid).await {
        error!("Failed to unpublish track for {device_id}: {error}");
    } else {
        info!("Unpublished stale track for device {device_id}");
    }
}

fn generate_livekit_publish_token(
    livekit_api_key: &str,
    livekit_api_secret: &str,
    livekit_room: &str,
    bridge_instance_id: &str,
) -> Result<String> {
    use livekit_api::access_token::{AccessToken, VideoGrants};

    let token = AccessToken::with_api_key(livekit_api_key, livekit_api_secret)
        .with_identity(bridge_instance_id)
        .with_grants(VideoGrants {
            room_join: true,
            room: livekit_room.to_string(),
            can_publish: true,
            can_subscribe: false,
            ..Default::default()
        })
        .to_jwt()
        .map_err(|error| anyhow!("Failed to generate LiveKit token: {error}"))?;

    Ok(token)
}
