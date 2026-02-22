use std::collections::VecDeque;
use std::time::Duration;

use async_nats::jetstream;
use futures_util::StreamExt;
use tracing::{error, info};
use webrtc_vad::{SampleRate, Vad, VadMode};

use crate::audio_chunk_capnp;
use crate::config::AppConfig;
use crate::shared::audio::utils::decode_flac_bytes_to_pcm_i16::decode_flac_bytes_to_pcm_i16;
use crate::shared::audio::utils::encode_pcm_i16_to_flac_bytes::encode_pcm_i16_to_flac_bytes;
use crate::shared::nats::types::audio_metadata::AudioMetadata;
use crate::shared::nats::utils::publish_transcription::publish_transcription;
use crate::shared::speaches::services::transcribe_audio::transcribe_audio;
use crate::shared::webrtc_vad::services::webrtc_vad_processor::WebRtcVadProcessor;
use crate::shared::webrtc_vad::states::speech_state::SpeechState;
use crate::shared::webrtc_vad::types::speech_segment::SpeechSegment;

struct DeserializedAudioChunk {
    audio_data: Vec<u8>,
    sample_rate_hz: u32,
    audio_format: String,
}

fn deserialize_audio_chunk(payload: &[u8]) -> Result<DeserializedAudioChunk, capnp::Error> {
    let reader =
        capnp::serialize::read_message_from_flat_slice(&mut &*payload, Default::default())?;
    let audio_chunk = reader.get_root::<audio_chunk_capnp::audio_chunk::Reader>()?;
    Ok(DeserializedAudioChunk {
        audio_data: audio_chunk.get_audio_data()?.to_vec(),
        sample_rate_hz: audio_chunk.get_sample_rate_hz(),
        audio_format: audio_chunk.get_audio_format()?.to_string()?,
    })
}

fn map_sample_rate_to_vad_rate(sample_rate_hz: u32) -> Option<SampleRate> {
    match sample_rate_hz {
        8000 => Some(SampleRate::Rate8kHz),
        16000 => Some(SampleRate::Rate16kHz),
        32000 => Some(SampleRate::Rate32kHz),
        48000 => Some(SampleRate::Rate48kHz),
        _ => None,
    }
}

fn convert_pcm_samples_to_le_bytes(pcm_samples: &[i16]) -> Vec<u8> {
    pcm_samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn spawn_transcription_task(
    config: &AppConfig,
    reqwest_client: &reqwest::Client,
    jetstream_context: &jetstream::Context,
    stream_id: &str,
    segment: SpeechSegment,
    sample_rate_hz: u32,
    audio_format: &str,
) {
    let flac_data = match encode_pcm_i16_to_flac_bytes(&segment.pcm_samples, sample_rate_hz, 1) {
        Ok(data) => data,
        Err(error) => {
            error!(
                "Failed to encode PCM to FLAC (stream_id={stream_id}, start={:.2}s, end={:.2}s): {error}",
                segment.start_s, segment.end_s
            );
            return;
        }
    };

    let audio_data_for_publish = flac_data.clone();
    let reqwest_client_clone = reqwest_client.clone();
    let config_clone = config.clone();
    let jetstream_context_clone = jetstream_context.clone();
    let stream_id_clone = stream_id.to_string();
    let audio_format_clone = audio_format.to_string();

    info!(
        "WebRTC VAD segment finalized; sending for transcription (stream_id={stream_id}, start={:.2}s, end={:.2}s, {} samples)",
        segment.start_s,
        segment.end_s,
        segment.pcm_samples.len()
    );

    tokio::spawn(async move {
        match transcribe_audio(
            &reqwest_client_clone,
            &config_clone.speaches_base_url,
            &flac_data,
            &config_clone.transcription_model,
            "audio.flac",
            "audio/flac",
        )
        .await
        {
            Ok(transcription_response) => {
                let timestamp_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

                let transcription_subject = format!(
                    "{}.{stream_id_clone}",
                    config_clone.transcription_subject_prefix
                );

                info!(
                    "Transcription completed (stream_id={stream_id_clone}, start={:.2}s, end={:.2}s): {}",
                    segment.start_s, segment.end_s, transcription_response.text
                );

                let audio_metadata = AudioMetadata {
                    sample_rate_hz,
                    audio_data: &audio_data_for_publish,
                    audio_format: &audio_format_clone,
                };

                if let Err(error) = publish_transcription(
                    &jetstream_context_clone,
                    &transcription_subject,
                    &stream_id_clone,
                    timestamp_ns,
                    &transcription_response,
                    &audio_metadata,
                )
                .await
                {
                    error!(
                        "Failed to publish transcription to NATS (stream_id={stream_id_clone}): {error}",
                    );
                }
            }
            Err(error) => {
                error!(
                    "Failed to transcribe WebRTC VAD audio segment (stream_id={stream_id_clone}, start={:.2}s, end={:.2}s): {error}",
                    segment.start_s, segment.end_s
                );
            }
        }
    });
}

pub async fn process_audio_stream_from_nats() -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::get();

    info!("Connecting to NATS at {}", config.nats_url);

    let nats_client = async_nats::connect(&config.nats_url).await?;
    let jetstream_context = jetstream::new(nats_client);

    info!(
        "Getting NATS stream: {} for subject filter: {}",
        config.nats_stream_name, config.subject_filter
    );

    let stream = jetstream_context
        .get_stream(&config.nats_stream_name)
        .await?;

    let consumer = stream
        .create_consumer(jetstream::consumer::pull::Config {
            durable_name: Some(config.queue_group.to_string()),
            filter_subject: config.subject_filter.to_string(),
            deliver_policy: jetstream::consumer::DeliverPolicy::New,
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        })
        .await?;

    info!(
        "Created consumer for stream: {} with subject filter: {}",
        config.nats_stream_name, config.subject_filter
    );

    let mut messages = consumer
        .stream()
        .max_messages_per_batch(100)
        .heartbeat(Duration::from_secs(5))
        .messages()
        .await?;

    let reqwest_client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            error!("Failed to create reqwest client: {error}");
            return Err(Box::new(error));
        }
    };

    let mut speech_state = SpeechState::new();
    let mut frame_buffer = VecDeque::new();
    let mut frame_byte_accumulator: Vec<u8> = Vec::new();
    let mut current_sample_rate_hz: u32 = 0;
    let mut current_audio_format: String = String::new();
    let mut last_stream_id: Option<String> = None;
    let mut frame_bytes: usize = 0;

    let parse_vad_mode = |mode: &str| -> VadMode {
        match mode {
            "Quality" => VadMode::Quality,
            "LowBitrate" => VadMode::LowBitrate,
            "Aggressive" => VadMode::Aggressive,
            "VeryAggressive" => VadMode::VeryAggressive,
            _ => VadMode::Aggressive,
        }
    };
    let mut webrtc_vad = Vad::new_with_rate_and_mode(
        SampleRate::Rate16kHz,
        parse_vad_mode(&config.webrtc_vad_mode),
    );

    info!("Starting to process audio messages with WebRTC VAD from NATS");

    loop {
        match messages.next().await {
            Some(Ok(message)) => {
                let payload = message.payload.to_vec();
                let message_subject = message.subject.to_string();

                // Extract stream_id from subject (e.g., "EMERGENCY_AUDIO_STREAMS.lincoln.fire" -> "lincoln.fire")
                let stream_id = message_subject
                    .split_once('.')
                    .map_or("", |(_, rest)| rest)
                    .to_string();
                last_stream_id = Some(stream_id.clone());

                let chunk = match deserialize_audio_chunk(&payload) {
                    Ok(result) => result,
                    Err(error) => {
                        error!("Failed to deserialize AudioChunk: {error}");
                        if let Err(error) = message.ack().await {
                            error!("Failed to acknowledge message: {error}");
                        }
                        continue;
                    }
                };

                current_audio_format = chunk.audio_format.clone();

                if current_audio_format != "flac" {
                    error!(
                        "Unsupported audio format: {}. Only flac is supported",
                        current_audio_format
                    );
                    if let Err(error) = message.ack().await {
                        error!("Failed to acknowledge message: {error}");
                    }
                    continue;
                }

                let (pcm_samples, decoded_sample_rate_hz) =
                    match decode_flac_bytes_to_pcm_i16(&chunk.audio_data) {
                        Ok(result) => result,
                        Err(error) => {
                            error!("Failed to decode FLAC chunk: {error}");
                            if let Err(error) = message.ack().await {
                                error!("Failed to acknowledge message: {error}");
                            }
                            continue;
                        }
                    };

                if decoded_sample_rate_hz != chunk.sample_rate_hz {
                    error!(
                        "Sample rate mismatch: FLAC header reports {decoded_sample_rate_hz} Hz but message metadata reports {} Hz, skipping chunk",
                        chunk.sample_rate_hz
                    );
                    if let Err(error) = message.ack().await {
                        error!("Failed to acknowledge message: {error}");
                    }
                    continue;
                }

                let pcm_le_bytes = convert_pcm_samples_to_le_bytes(&pcm_samples);

                if chunk.sample_rate_hz != current_sample_rate_hz {
                    if current_sample_rate_hz != 0 {
                        info!(
                            "Sample rate changed from {current_sample_rate_hz} to {}, reinitializing VAD",
                            chunk.sample_rate_hz
                        );
                        frame_byte_accumulator.clear();
                        let (new_speech_state, new_frame_buffer) =
                            WebRtcVadProcessor::create_initial_state();
                        speech_state = new_speech_state;
                        frame_buffer = new_frame_buffer;
                    }

                    current_sample_rate_hz = chunk.sample_rate_hz;
                    let vad_rate = match map_sample_rate_to_vad_rate(current_sample_rate_hz) {
                        Some(rate) => rate,
                        None => {
                            error!(
                                "Unsupported sample rate for WebRTC VAD: {}. Supported: 8000, 16000, 32000, 48000",
                                current_sample_rate_hz
                            );
                            if let Err(error) = message.ack().await {
                                error!("Failed to acknowledge message: {error}");
                            }
                            continue;
                        }
                    };
                    webrtc_vad = Vad::new_with_rate_and_mode(
                        vad_rate,
                        parse_vad_mode(&config.webrtc_vad_mode),
                    );
                    let samples_per_frame =
                        WebRtcVadProcessor::samples_per_frame(config, current_sample_rate_hz);
                    frame_bytes = samples_per_frame * 2;
                    frame_byte_accumulator = Vec::with_capacity(frame_bytes * 4);
                    info!(
                        "VAD initialized at {current_sample_rate_hz} Hz (frame_bytes={frame_bytes})"
                    );
                }

                frame_byte_accumulator.extend_from_slice(&pcm_le_bytes);

                while frame_byte_accumulator.len() >= frame_bytes {
                    let frame_bytes_data = frame_byte_accumulator
                        .drain(0..frame_bytes)
                        .collect::<Vec<u8>>();

                    let samples: Vec<i16> = frame_bytes_data
                        .chunks_exact(2)
                        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();

                    let result = WebRtcVadProcessor::process_frame(
                        config,
                        current_sample_rate_hz,
                        &mut webrtc_vad,
                        speech_state,
                        frame_buffer,
                        &samples,
                    );

                    speech_state = result.speech_state;
                    frame_buffer = result.frame_buffer;

                    for segment in result.segments {
                        spawn_transcription_task(
                            config,
                            &reqwest_client,
                            &jetstream_context,
                            &stream_id,
                            segment,
                            current_sample_rate_hz,
                            &current_audio_format,
                        );
                    }
                }

                if WebRtcVadProcessor::should_reset_due_to_inactivity(&speech_state, 60) {
                    let (new_speech_state, new_frame_buffer) =
                        WebRtcVadProcessor::create_initial_state();
                    speech_state = new_speech_state;
                    frame_buffer = new_frame_buffer;
                    info!("WebRTC VAD processor reset due to inactivity");
                }

                if let Err(error) = message.ack().await {
                    error!("Failed to acknowledge message: {error}");
                }
            }
            Some(Err(error)) => {
                error!("Error receiving message: {error}");
                break;
            }
            None => {
                info!("Stream ended");
                break;
            }
        }
    }

    if let Some(final_segment) =
        WebRtcVadProcessor::finalize(config, current_sample_rate_hz, speech_state)
    {
        match &last_stream_id {
            Some(stream_id) => {
                info!(
                    "Final WebRTC VAD segment (stream_id={stream_id}, start={:.2}s, end={:.2}s, {} samples)",
                    final_segment.start_s,
                    final_segment.end_s,
                    final_segment.pcm_samples.len()
                );
                spawn_transcription_task(
                    config,
                    &reqwest_client,
                    &jetstream_context,
                    stream_id,
                    final_segment,
                    current_sample_rate_hz,
                    &current_audio_format,
                );
            }
            None => {
                info!(
                    "Skipping final WebRTC VAD segment, no stream_id available (start={:.2}s, end={:.2}s, {} samples)",
                    final_segment.start_s,
                    final_segment.end_s,
                    final_segment.pcm_samples.len()
                );
            }
        }
    }

    Ok(())
}
