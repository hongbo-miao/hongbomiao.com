use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use futures::StreamExt;
use livekit::prelude::*;
use livekit_api::access_token::{AccessToken, VideoGrants};
use prost::Message;
use pulsar::consumer::InitialPosition;
use pulsar::{
    Consumer, ConsumerOptions, DeserializeMessage, Payload, Producer, Pulsar, SerializeMessage,
    SubType, TokioExecutor, producer,
};
use serde_json::json;
use sherpa_onnx::{
    OfflineCohereTranscribeModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineTransducerModelConfig, VadModelConfig, VoiceActivityDetector,
};
use tracing::{error, info};

use crate::shared::audio::services::create_online_recognizer::create_online_recognizer;
use crate::shared::audio::types::audio_chunk::AudioChunk;

static TRANSCRIPT_AVRO_SCHEMA: LazyLock<apache_avro::Schema> = LazyLock::new(|| {
    apache_avro::Schema::parse_str(include_str!("../../../../../schemas/audio_transcript.avsc"))
        .expect("Failed to parse audio transcript Avro schema")
});

const PCM_SAMPLE_RATE_HZ: u32 = 16_000;
const STALE_CHUNK_THRESHOLD_MS: u128 = 2_000;
const OPUS_FRAME_SAMPLE_COUNT: usize = 320;
const DEVICE_STALE_TIMEOUT_S: u64 = 5;
const AUDIO_CHANNEL_BUFFER: usize = 512;
const ASR_NUM_THREADS: i32 = 2;
const VAD_WINDOW_SIZE: usize = 256;
const VAD_BUFFER_DURATION_S: f32 = 60.0;
const VAD_BUFFER_TRIM_WINDOW_COUNT: usize = 10;
const VAD_SPEECH_PAD_HEAD_SAMPLES: usize = PCM_SAMPLE_RATE_HZ as usize * 200 / 1000; // 0.2s head padding
const VAD_THRESHOLD: f32 = 0.4;
const VAD_MIN_SILENCE_DURATION_S: f32 = 0.08;
const VAD_MIN_SPEECH_DURATION_S: f32 = 0.1;
const VAD_MAX_SPEECH_DURATION_S: f32 = 15.0;
// Keep the last 500ms of chunks unacked so the broker redelivers them as acoustic pre-roll
// if this pod crashes mid-word, preventing the ASR model from receiving a truncated utterance.
const ACK_DELAY_MS: u64 = 500;
const LIVEKIT_RECONNECT_DELAY_S: u64 = 2;
const MAX_LIVEKIT_PUBLISH_RETRIES: u32 = 3;

struct RawPayload;

impl DeserializeMessage for RawPayload {
    type Output = Vec<u8>;

    fn deserialize_message(payload: &Payload) -> Self::Output {
        payload.data.clone()
    }
}

struct TranscriptMessage {
    payload: Vec<u8>,
    partition_key: String,
}

impl SerializeMessage for TranscriptMessage {
    fn serialize_message(input: Self) -> Result<producer::Message, pulsar::Error> {
        Ok(producer::Message {
            payload: input.payload,
            partition_key: Some(input.partition_key),
            ..Default::default()
        })
    }
}

struct TranscriptSegment {
    device_id: String,
    text: String,
    timestamp_ns: i64,
    is_final: bool,
}

#[allow(clippy::too_many_arguments)]
pub async fn transcribe_audio_stream(
    pulsar_url: &str,
    pulsar_topic: &str,
    pulsar_output_topic: &str,
    livekit_url: &str,
    livekit_api_key: &str,
    livekit_api_secret: &str,
    livekit_room: &str,
    instance_id: &str,
    asr_model_type: &str,
    cohere_transcribe_model_dir: &str,
    parakeet_tdt_model_dir: &str,
    ten_vad_model_dir: &str,
    zipformer_model_dir: &str,
) -> Result<()> {
    let pulsar_client: Pulsar<TokioExecutor> =
        Pulsar::builder(pulsar_url, TokioExecutor).build().await?;

    let subscription_name = "audio-stream-transcriber-subscription";
    let mut consumer: Consumer<RawPayload, TokioExecutor> = pulsar_client
        .consumer()
        .with_topic(pulsar_topic)
        .with_subscription(subscription_name)
        .with_subscription_type(SubType::KeyShared)
        .with_consumer_name(instance_id)
        .with_options(ConsumerOptions {
            initial_position: InitialPosition::Latest,
            ..Default::default()
        })
        .build()
        .await?;

    let transcript_producer: Producer<TokioExecutor> = pulsar_client
        .producer()
        .with_topic(pulsar_output_topic)
        .build()
        .await?;

    let room = connect_to_livekit_room(
        livekit_url,
        livekit_api_key,
        livekit_api_secret,
        livekit_room,
        instance_id,
    )
    .await?;
    let (transcript_sender, mut transcript_receiver) =
        tokio::sync::mpsc::unbounded_channel::<TranscriptSegment>();

    let livekit_url_for_reconnect = livekit_url.to_string();
    let livekit_api_key_for_reconnect = livekit_api_key.to_string();
    let livekit_api_secret_for_reconnect = livekit_api_secret.to_string();
    let livekit_room_for_reconnect = livekit_room.to_string();
    let instance_id_for_reconnect = instance_id.to_string();
    let mut transcript_producer = transcript_producer;
    tokio::spawn(async move {
        let mut room = room;
        while let Some(segment) = transcript_receiver.recv().await {
            let mut attempt: u32 = 0;
            loop {
                match publish_transcript_to_livekit(&room, &segment).await {
                    Ok(()) => break,
                    Err(error) => {
                        if attempt < MAX_LIVEKIT_PUBLISH_RETRIES {
                            attempt += 1;
                            error!(
                                "LiveKit publish failed (attempt {attempt}/{MAX_LIVEKIT_PUBLISH_RETRIES}), reconnecting: {error}"
                            );
                            match connect_to_livekit_room(
                                &livekit_url_for_reconnect,
                                &livekit_api_key_for_reconnect,
                                &livekit_api_secret_for_reconnect,
                                &livekit_room_for_reconnect,
                                &instance_id_for_reconnect,
                            )
                            .await
                            {
                                Ok(new_room) => {
                                    room = new_room;
                                }
                                Err(reconnect_error) => {
                                    error!("Failed to reconnect to LiveKit: {reconnect_error}");
                                    tokio::time::sleep(Duration::from_secs(
                                        LIVEKIT_RECONNECT_DELAY_S,
                                    ))
                                    .await;
                                }
                            }
                        } else {
                            error!(
                                "Dropping transcript segment after {MAX_LIVEKIT_PUBLISH_RETRIES} LiveKit retries: {error}"
                            );
                            break;
                        }
                    }
                }
            }
            if let Err(error) =
                publish_transcript_to_pulsar(&mut transcript_producer, &segment).await
            {
                error!("Failed to publish transcript segment to Pulsar: {error}");
            }
        }
    });

    let mut device_audio_senders: HashMap<String, mpsc::SyncSender<Vec<f32>>> = HashMap::new();
    let mut device_opus_decoders: HashMap<String, opus::Decoder> = HashMap::new();
    let mut ack_delay_queue = VecDeque::new();

    while let Some(message_result) = consumer.next().await {
        let ack_cutoff = Instant::now() - Duration::from_millis(ACK_DELAY_MS);
        while let Some((received_at, _)) = ack_delay_queue.front() {
            if *received_at <= ack_cutoff {
                let (_, queued_message) = ack_delay_queue.pop_front().expect("front exists");
                consumer.ack(&queued_message).await.ok();
            } else {
                break;
            }
        }

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

        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let age_ms = now_ns.saturating_sub(audio_chunk.timestamp_ns as u128) / 1_000_000;
        if age_ms > STALE_CHUNK_THRESHOLD_MS {
            consumer.ack(&pulsar_message).await.ok();
            continue;
        }

        let device_id = audio_chunk.device_id.clone();

        if !device_audio_senders.contains_key(&device_id) {
            let (audio_sender, audio_receiver) =
                mpsc::sync_channel::<Vec<f32>>(AUDIO_CHANNEL_BUFFER);

            let transcript_sender_clone = transcript_sender.clone();
            let device_id_clone = device_id.clone();
            let asr_model_type_clone = asr_model_type.to_string();
            let cohere_transcribe_model_dir_clone = cohere_transcribe_model_dir.to_string();
            let parakeet_tdt_model_dir_clone = parakeet_tdt_model_dir.to_string();
            let ten_vad_model_dir_clone = ten_vad_model_dir.to_string();
            let zipformer_model_dir_clone = zipformer_model_dir.to_string();

            std::thread::spawn(move || {
                run_vad_thread(
                    device_id_clone,
                    audio_receiver,
                    transcript_sender_clone,
                    asr_model_type_clone,
                    cohere_transcribe_model_dir_clone,
                    parakeet_tdt_model_dir_clone,
                    ten_vad_model_dir_clone,
                    zipformer_model_dir_clone,
                );
            });

            device_audio_senders.insert(device_id.clone(), audio_sender);

            match opus::Decoder::new(PCM_SAMPLE_RATE_HZ, opus::Channels::Mono) {
                Ok(decoder) => {
                    device_opus_decoders.insert(device_id.clone(), decoder);
                }
                Err(error) => {
                    error!("Failed to create Opus decoder for {device_id}: {error}");
                    device_audio_senders.remove(&device_id);
                    consumer.ack(&pulsar_message).await.ok();
                    continue;
                }
            }
        }

        let decoder = match device_opus_decoders.get_mut(&device_id) {
            Some(decoder) => decoder,
            None => {
                consumer.ack(&pulsar_message).await.ok();
                continue;
            }
        };

        let mut pcm_buffer = vec![0i16; OPUS_FRAME_SAMPLE_COUNT];
        let mut f32_samples: Vec<f32> = Vec::new();

        for opus_packet in &audio_chunk.opus_packets {
            match decoder.decode(opus_packet, &mut pcm_buffer, false) {
                Ok(decoded_sample_count) => {
                    f32_samples.extend(
                        pcm_buffer[..decoded_sample_count]
                            .iter()
                            .map(|&sample| sample as f32 / i16::MAX as f32),
                    );
                }
                Err(error) => {
                    error!("Failed to decode Opus packet for {device_id}: {error}");
                }
            }
        }

        if !f32_samples.is_empty()
            && let Some(sender) = device_audio_senders.get(&device_id)
        {
            match sender.try_send(f32_samples) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(_)) => {
                    error!("Audio channel full for {device_id}, dropping audio chunk");
                    consumer.ack(&pulsar_message).await.ok();
                    continue;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    device_audio_senders.remove(&device_id);
                    device_opus_decoders.remove(&device_id);
                }
            }
        }

        ack_delay_queue.push_back((Instant::now(), pulsar_message));
    }

    for (_, queued_message) in ack_delay_queue {
        consumer.ack(&queued_message).await.ok();
    }

    Ok(())
}

fn build_cohere_transcribe_offline_recognizer_config(
    cohere_transcribe_model_dir: &str,
) -> OfflineRecognizerConfig {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.cohere_transcribe = OfflineCohereTranscribeModelConfig {
        encoder: Some(format!("{cohere_transcribe_model_dir}/encoder.int8.onnx")),
        decoder: Some(format!("{cohere_transcribe_model_dir}/decoder.int8.onnx")),
        use_punct: true,
        use_itn: true,
        ..Default::default()
    };
    config.model_config.tokens = Some(format!("{cohere_transcribe_model_dir}/tokens.txt"));
    config.model_config.num_threads = ASR_NUM_THREADS;
    config
}

fn build_parakeet_tdt_offline_recognizer_config(
    parakeet_tdt_model_dir: &str,
) -> OfflineRecognizerConfig {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(format!("{parakeet_tdt_model_dir}/encoder.int8.onnx")),
        decoder: Some(format!("{parakeet_tdt_model_dir}/decoder.int8.onnx")),
        joiner: Some(format!("{parakeet_tdt_model_dir}/joiner.int8.onnx")),
    };
    config.model_config.tokens = Some(format!("{parakeet_tdt_model_dir}/tokens.txt"));
    config.model_config.model_type = Some("nemo_transducer".to_string());
    config.model_config.num_threads = ASR_NUM_THREADS;
    config
}

fn build_vad_config(ten_vad_model_dir: &str) -> VadModelConfig {
    let mut config = VadModelConfig::default();
    config.ten_vad.model = Some(format!("{ten_vad_model_dir}/ten_vad.onnx"));
    config.ten_vad.threshold = VAD_THRESHOLD;
    config.ten_vad.min_silence_duration = VAD_MIN_SILENCE_DURATION_S;
    config.ten_vad.min_speech_duration = VAD_MIN_SPEECH_DURATION_S;
    config.ten_vad.max_speech_duration = VAD_MAX_SPEECH_DURATION_S;
    config.ten_vad.window_size = VAD_WINDOW_SIZE as i32;
    config.sample_rate = PCM_SAMPLE_RATE_HZ as i32;
    config
}

fn run_asr_thread(
    device_id: String,
    asr_receiver: mpsc::Receiver<Vec<f32>>,
    recognizer: OfflineRecognizer,
    asr_model_type: String,
    transcript_sender: tokio::sync::mpsc::UnboundedSender<TranscriptSegment>,
) {
    for samples in asr_receiver {
        let stream = recognizer.create_stream();
        if asr_model_type == "cohere-transcribe" {
            stream.set_option("language", "en");
        }
        stream.accept_waveform(PCM_SAMPLE_RATE_HZ as i32, &samples);
        recognizer.decode(&stream);
        if let Some(result) = stream.get_result() {
            let text = result.text.trim().to_string();
            if !text.is_empty() {
                info!("Transcript: device={device_id} text={text}");
                let timestamp_ns = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as i64;
                if transcript_sender
                    .send(TranscriptSegment {
                        device_id: device_id.clone(),
                        text,
                        timestamp_ns,
                        is_final: true,
                    })
                    .is_err()
                {
                    return;
                }
            }
        }
    }
}

enum ZipformerMessage {
    Samples(u64, Vec<f32>),
}

fn run_zipformer_thread(
    device_id: String,
    zipformer_receiver: mpsc::Receiver<ZipformerMessage>,
    transcript_sender: tokio::sync::mpsc::UnboundedSender<TranscriptSegment>,
    zipformer_model_dir: String,
) {
    let recognizer = create_online_recognizer(&zipformer_model_dir);
    let stream = recognizer.create_stream();
    let mut last_interim_text = String::new();
    let mut current_segment_id: u64 = 0;

    for message in zipformer_receiver {
        match message {
            ZipformerMessage::Samples(segment_id, samples) => {
                if segment_id != current_segment_id {
                    recognizer.reset(&stream);
                    last_interim_text.clear();
                    current_segment_id = segment_id;
                }
                stream.accept_waveform(PCM_SAMPLE_RATE_HZ as i32, &samples);
                while recognizer.is_ready(&stream) {
                    recognizer.decode(&stream);
                }
                if let Some(result) = recognizer.get_result(&stream) {
                    // Zipformer outputs all-uppercase; lowercase for readability as interim text.
                    let partial_text = result.text.trim().to_lowercase();
                    if !partial_text.is_empty() && partial_text != last_interim_text {
                        last_interim_text = partial_text.clone();
                        let timestamp_ns = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as i64;
                        transcript_sender
                            .send(TranscriptSegment {
                                device_id: device_id.clone(),
                                text: partial_text,
                                timestamp_ns,
                                is_final: false,
                            })
                            .ok();
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_vad_thread(
    device_id: String,
    audio_receiver: mpsc::Receiver<Vec<f32>>,
    transcript_sender: tokio::sync::mpsc::UnboundedSender<TranscriptSegment>,
    asr_model_type: String,
    cohere_transcribe_model_dir: String,
    parakeet_tdt_model_dir: String,
    ten_vad_model_dir: String,
    zipformer_model_dir: String,
) {
    let offline_recognizer_config = match asr_model_type.as_str() {
        "cohere-transcribe" => {
            build_cohere_transcribe_offline_recognizer_config(&cohere_transcribe_model_dir)
        }
        "parakeet-tdt" => build_parakeet_tdt_offline_recognizer_config(&parakeet_tdt_model_dir),
        unknown => {
            error!("Unknown ASR_MODEL_TYPE '{unknown}' for device {device_id}");
            return;
        }
    };
    let recognizer = match OfflineRecognizer::create(&offline_recognizer_config) {
        Some(recognizer) => recognizer,
        None => {
            error!("Failed to create recognizer for device {device_id}");
            return;
        }
    };
    let vad = match VoiceActivityDetector::create(
        &build_vad_config(&ten_vad_model_dir),
        VAD_BUFFER_DURATION_S,
    ) {
        Some(vad) => vad,
        None => {
            error!("Failed to create VAD for device {device_id}");
            return;
        }
    };

    // Blocks VAD (rather than dropping) when ASR falls behind; sized to 15 segments × 15s max each.
    let (asr_sender, asr_receiver) = mpsc::sync_channel::<Vec<f32>>(15);
    // Drop interim-text samples rather than blocking the VAD thread if Zipformer falls behind.
    let (zipformer_sender, zipformer_receiver) =
        mpsc::sync_channel::<ZipformerMessage>(AUDIO_CHANNEL_BUFFER);

    let device_id_for_asr = device_id.clone();
    let asr_model_type_for_asr = asr_model_type.clone();
    let transcript_sender_for_asr = transcript_sender.clone();
    std::thread::spawn(move || {
        run_asr_thread(
            device_id_for_asr,
            asr_receiver,
            recognizer,
            asr_model_type_for_asr,
            transcript_sender_for_asr,
        );
    });

    let device_id_for_zipformer = device_id.clone();
    let transcript_sender_for_zipformer = transcript_sender.clone();
    std::thread::spawn(move || {
        run_zipformer_thread(
            device_id_for_zipformer,
            zipformer_receiver,
            transcript_sender_for_zipformer,
            zipformer_model_dir,
        );
    });

    let mut buffer: Vec<f32> = Vec::new();
    let mut vad_offset: usize = 0;
    let mut pre_speech_ring: VecDeque<f32> = VecDeque::with_capacity(VAD_SPEECH_PAD_HEAD_SAMPLES);
    let mut zipformer_segment_id: u64 = 0;

    loop {
        match audio_receiver.recv_timeout(Duration::from_secs(DEVICE_STALE_TIMEOUT_S)) {
            Ok(samples) => {
                if !vad.detected() {
                    pre_speech_ring.extend(samples.iter().copied());
                    if pre_speech_ring.len() > VAD_SPEECH_PAD_HEAD_SAMPLES {
                        let drain_count = pre_speech_ring.len() - VAD_SPEECH_PAD_HEAD_SAMPLES;
                        pre_speech_ring.drain(..drain_count);
                    }
                }

                zipformer_sender
                    .try_send(ZipformerMessage::Samples(
                        zipformer_segment_id,
                        samples.clone(),
                    ))
                    .ok();

                buffer.extend_from_slice(&samples);

                while vad_offset + VAD_WINDOW_SIZE <= buffer.len() {
                    vad.accept_waveform(&buffer[vad_offset..vad_offset + VAD_WINDOW_SIZE]);
                    vad_offset += VAD_WINDOW_SIZE;
                }

                if !vad.detected() && buffer.len() > VAD_BUFFER_TRIM_WINDOW_COUNT * VAD_WINDOW_SIZE
                {
                    let trim_count = buffer.len() - VAD_BUFFER_TRIM_WINDOW_COUNT * VAD_WINDOW_SIZE;
                    vad_offset = vad_offset.saturating_sub(trim_count);
                    buffer.drain(..trim_count);
                }

                if !vad.is_empty() {
                    while !vad.is_empty() {
                        if let Some(segment) = vad.front() {
                            let padded: Vec<f32> = pre_speech_ring
                                .iter()
                                .copied()
                                .chain(segment.samples().iter().copied())
                                .collect();
                            zipformer_segment_id += 1;
                            if asr_sender.send(padded).is_err() {
                                return;
                            }
                        }
                        vad.pop();
                        pre_speech_ring.clear();
                    }
                    buffer.clear();
                    vad_offset = 0;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                vad.flush();
                while !vad.is_empty() {
                    if let Some(segment) = vad.front() {
                        let padded: Vec<f32> = pre_speech_ring
                            .iter()
                            .copied()
                            .chain(segment.samples().iter().copied())
                            .collect();
                        asr_sender.send(padded).ok();
                    }
                    vad.pop();
                    pre_speech_ring.clear();
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

async fn publish_transcript_to_livekit(room: &Room, segment: &TranscriptSegment) -> Result<()> {
    let livekit_bytes = json!({
        "device_id": segment.device_id,
        "text": segment.text,
        "timestamp_ns": segment.timestamp_ns,
        "is_final": segment.is_final,
    })
    .to_string()
    .into_bytes();

    room.local_participant()
        .publish_data(DataPacket {
            payload: livekit_bytes,
            topic: Some("transcript".to_string()),
            reliable: true,
            destination_identities: vec![],
        })
        .await
        .map_err(|error| anyhow!("Failed to publish transcript to LiveKit: {error}"))
}

async fn publish_transcript_to_pulsar(
    transcript_producer: &mut Producer<TokioExecutor>,
    segment: &TranscriptSegment,
) -> Result<()> {
    if !segment.is_final {
        return Ok(());
    }

    let mut avro_record = apache_avro::types::Record::new(&TRANSCRIPT_AVRO_SCHEMA)
        .expect("Failed to create Avro record");
    avro_record.put("device_id", segment.device_id.clone());
    avro_record.put("text", segment.text.clone());
    avro_record.put("timestamp_ns", segment.timestamp_ns);
    let pulsar_bytes = apache_avro::to_avro_datum(&TRANSCRIPT_AVRO_SCHEMA, avro_record)?;

    transcript_producer
        .send_non_blocking(TranscriptMessage {
            payload: pulsar_bytes,
            partition_key: segment.device_id.clone(),
        })
        .await
        .map_err(|error| anyhow!("Failed to send transcript to Pulsar: {error}"))?
        .await
        .map_err(|error| anyhow!("Failed to receive Pulsar send acknowledgement: {error}"))?;

    Ok(())
}

async fn connect_to_livekit_room(
    livekit_url: &str,
    livekit_api_key: &str,
    livekit_api_secret: &str,
    livekit_room: &str,
    instance_id: &str,
) -> Result<Room> {
    let token = generate_livekit_token(
        livekit_api_key,
        livekit_api_secret,
        livekit_room,
        instance_id,
    )?;
    let (room, _room_event_receiver) = Room::connect(livekit_url, &token, RoomOptions::default())
        .await
        .map_err(|error| anyhow!("Failed to connect to LiveKit room: {error}"))?;
    Ok(room)
}

fn generate_livekit_token(
    livekit_api_key: &str,
    livekit_api_secret: &str,
    livekit_room: &str,
    instance_id: &str,
) -> Result<String> {
    let token = AccessToken::with_api_key(livekit_api_key, livekit_api_secret)
        .with_identity(instance_id)
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
