use crate::config::AppConfig;
use crate::shared::speaches::services::transcription_service::make_transcription_request;
use crate::shared::webrtc_vad::services::webrtc_vad_processor::WebRtcVadProcessor;
use crate::shared::webrtc_vad::states::speech_state::SpeechState;
use async_nats::jetstream;
use futures_util::StreamExt;
use std::collections::VecDeque;
use std::time::Duration;
use tracing::{error, info};
use webrtc_vad::{SampleRate, Vad, VadMode};

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
    let samples_per_chunk = WebRtcVadProcessor::samples_per_frame(config);
    let frame_bytes = samples_per_chunk * 2;

    let mut frame_byte_accumulator: Vec<u8> = Vec::with_capacity(frame_bytes * 4);

    let webrtc_vad_mode = match config.webrtc_vad_mode.as_str() {
        "Quality" => VadMode::Quality,
        "LowBitrate" => VadMode::LowBitrate,
        "Aggressive" => VadMode::Aggressive,
        "VeryAggressive" => VadMode::VeryAggressive,
        _ => VadMode::Aggressive,
    };
    let mut webrtc_vad = Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, webrtc_vad_mode);

    info!("Starting to process audio messages with WebRTC VAD from NATS");

    loop {
        match messages.next().await {
            Some(Ok(message)) => {
                let payload = message.payload.to_vec();

                if payload.len() < 4 {
                    error!("Received invalid message (too short)");
                    if let Err(error) = message.ack().await {
                        error!("Failed to acknowledge message: {error}");
                    }
                    continue;
                }

                let frame_length =
                    u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;

                if payload.len() < 4 + frame_length {
                    error!("Received invalid message (payload shorter than declared length)");
                    if let Err(error) = message.ack().await {
                        error!("Failed to acknowledge message: {error}");
                    }
                    continue;
                }

                let pcm_data = &payload[4..4 + frame_length];

                frame_byte_accumulator.extend_from_slice(pcm_data);

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
                        &mut webrtc_vad,
                        speech_state,
                        frame_buffer,
                        &samples,
                    );

                    speech_state = result.speech_state;
                    frame_buffer = result.frame_buffer;

                    for segment in result.segments {
                        let reqwest_client_clone = reqwest_client.clone();
                        let config_clone = config.clone();

                        info!(
                            start_time = %segment.start,
                            end_time = %segment.end,
                            wav_size = segment.audio_data.len(),
                            "WebRTC VAD segment finalized; sending for transcription"
                        );

                        tokio::spawn(async move {
                            match make_transcription_request(
                                &reqwest_client_clone,
                                &config_clone.speaches_base_url,
                                &segment.audio_data,
                                &config_clone.transcription_model,
                            )
                            .await
                            {
                                Ok(transcription_text) => {
                                    info!(
                                        transcription = %transcription_text,
                                        start_time = %segment.start,
                                        end_time = %segment.end,
                                        "Transcription completed"
                                    );
                                }
                                Err(error) => {
                                    error!(
                                        "Failed to transcribe WebRTC VAD audio segment (start={}, end={}): {}",
                                        segment.start, segment.end, error
                                    );
                                }
                            }
                        });
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

    if let Some(final_segment) = WebRtcVadProcessor::finalize(config, speech_state) {
        let reqwest_client_clone = reqwest_client.clone();
        let config_clone = config.clone();

        info!(
            start_time = %final_segment.start,
            end_time = %final_segment.end,
            wav_size = final_segment.audio_data.len(),
            "Final WebRTC VAD segment; sending for transcription"
        );

        tokio::spawn(async move {
            match make_transcription_request(
                &reqwest_client_clone,
                &config_clone.speaches_base_url,
                &final_segment.audio_data,
                &config_clone.transcription_model,
            )
            .await
            {
                Ok(transcription_text) => {
                    info!(
                        transcription = %transcription_text,
                        start_time = %final_segment.start,
                        end_time = %final_segment.end,
                        "Final transcription completed"
                    );
                }
                Err(error) => {
                    error!(
                        "Failed to transcribe final WebRTC VAD audio segment (start={}, end={}): {}",
                        final_segment.start, final_segment.end, error
                    );
                }
            }
        });
    }

    Ok(())
}
