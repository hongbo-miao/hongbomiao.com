use ffmpeg_sidecar::command::FfmpegCommand;
use std::{collections::VecDeque, time::Duration};
use tokio::io::AsyncReadExt;
use tokio::process::ChildStdout;
use tokio::sync::broadcast;
use tracing::{error, info};
use webrtc_vad::{SampleRate, Vad, VadMode};

use crate::config::AppConfig;
use crate::shared::police_audio_stream::utils::spawn_transcription_and_broadcast::spawn_transcription_and_broadcast;
use crate::shared::webrtc_vad::services::webrtc_vad_processor::WebRtcVadProcessor;
use crate::shared::webrtc_vad::states::speech_state::SpeechState;

pub async fn process_police_audio_stream(
    police_stream_id: String,
    stream_url: String,
    audio_sender: broadcast::Sender<Vec<u8>>,
    server_sent_event_sender: broadcast::Sender<String>,
) {
    let config = AppConfig::get();

    info!(
        "Starting police audio stream processing for {}",
        police_stream_id
    );

    loop {
        let mut ffmpeg_child = match FfmpegCommand::new()
            .input(&stream_url)
            .arg("-nostdin")
            .hide_banner()
            .args(["-loglevel", "quiet"])
            .filter_complex(
                "pan=mono|c0=0.707107*FL+0.707107*FR,aresample=resampler=soxr:sample_rate=16000",
            )
            .format("s16le")
            .pipe_stdout()
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                error!("Failed to spawn ffmpeg for {police_stream_id}: {error}");
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        let stdout = ffmpeg_child
            .as_inner_mut()
            .stdout
            .take()
            .expect("ffmpeg child process did not have a stdout pipe");

        let mut stdout =
            ChildStdout::from_std(stdout).expect("Failed to convert to tokio ChildStdout");

        // Reuse a single reqwest client for all transcription calls
        let reqwest_client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
        {
            Ok(client) => client,
            Err(_) => {
                error!("Failed to create reqwest client for {}", police_stream_id);
                continue;
            }
        };

        // Initialize WebRTC VAD processor
        let mut speech_state = SpeechState::new();
        let mut frame_buffer = VecDeque::new();
        let samples_per_chunk = WebRtcVadProcessor::samples_per_frame(config);
        let frame_bytes = samples_per_chunk * 2; // s16le

        // Byte accumulator to split arbitrary read sizes into exact 20ms frames
        let mut frame_byte_accumulator: Vec<u8> = Vec::with_capacity(frame_bytes * 4);

        loop {
            let mut buffer = vec![0u8; 6400];
            let mut filled = 0usize;
            while filled < buffer.len() {
                match stdout.read(&mut buffer[filled..]).await {
                    Ok(0) => {
                        filled = 0;
                        break;
                    }
                    Ok(bytes_read) => {
                        filled += bytes_read;
                    }
                    Err(_) => {
                        filled = 0;
                        break;
                    }
                }
            }
            if filled == 0 {
                break;
            }

            // Broadcast raw audio chunk as-is for clients
            let _ = audio_sender.send(buffer.clone());

            // Feed into frame byte accumulator and process WebRTC VAD per 20ms frame
            frame_byte_accumulator.extend_from_slice(&buffer[..filled]);

            while frame_byte_accumulator.len() >= frame_bytes {
                let frame_bytes_data = frame_byte_accumulator
                    .drain(0..frame_bytes)
                    .collect::<Vec<u8>>();
                // Convert frame to i16 samples for WebRTC VAD processor
                let samples: Vec<i16> = frame_bytes_data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                // Create Vad instance for this frame to avoid Send issues across await boundaries
                let webrtc_vad_mode = match config.webrtc_vad_mode.as_str() {
                    "Quality" => VadMode::Quality,
                    "LowBitrate" => VadMode::LowBitrate,
                    "Aggressive" => VadMode::Aggressive,
                    "VeryAggressive" => VadMode::VeryAggressive,
                    _ => VadMode::Aggressive,
                };
                let mut webrtc_vad =
                    Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, webrtc_vad_mode);

                // Process frame with WebRTC VAD processor
                let result = WebRtcVadProcessor::process_frame(
                    config,
                    &mut webrtc_vad,
                    speech_state,
                    frame_buffer,
                    &samples,
                );

                // Update state for next iteration
                speech_state = result.speech_state;
                frame_buffer = result.frame_buffer;

                // Handle any completed speech segments
                for segment in result.segments {
                    let server_sent_event_sender_clone = server_sent_event_sender.clone();
                    let stream_id_clone = police_stream_id.clone();
                    let config_clone = config.clone();
                    let reqwest_client_clone = reqwest_client.clone();

                    info!(
                        police_stream_id = %stream_id_clone,
                        start_time = %segment.start,
                        end_time = %segment.end,
                        wav_size = segment.audio_data.len(),
                        "WebRTC VAD segment finalized; sending for transcription"
                    );

                    spawn_transcription_and_broadcast(
                        reqwest_client_clone,
                        &config_clone,
                        server_sent_event_sender_clone,
                        stream_id_clone,
                        segment.audio_data.clone(),
                    );
                }
            }

            // safeguard to avoid runaway - reset WebRTC VAD processor if inactive too long
            if WebRtcVadProcessor::should_reset_due_to_inactivity(&speech_state, 600) {
                // 10 minutes
                let (new_speech_state, new_frame_buffer) =
                    WebRtcVadProcessor::create_initial_state();
                speech_state = new_speech_state;
                frame_buffer = new_frame_buffer;
                info!(police_stream_id = %police_stream_id, "WebRTC VAD processor reset due to inactivity");
            }
        }

        // Handle any final segment when stream ends
        if let Some(final_segment) = WebRtcVadProcessor::finalize(config, speech_state) {
            let server_sent_event_sender_clone = server_sent_event_sender.clone();
            let stream_id_clone = police_stream_id.clone();
            let config_clone = config.clone();
            let reqwest_client_clone = reqwest_client.clone();

            info!(
                police_stream_id = %stream_id_clone,
                start_time = %final_segment.start,
                end_time = %final_segment.end,
                wav_size = final_segment.audio_data.len(),
                "Final WebRTC VAD segment; sending for transcription"
            );

            spawn_transcription_and_broadcast(
                reqwest_client_clone,
                &config_clone,
                server_sent_event_sender_clone,
                stream_id_clone,
                final_segment.audio_data.clone(),
            );
        }

        // Clean up the process
        if let Err(error) = ffmpeg_child.kill() {
            error!("Failed to kill ffmpeg process for {police_stream_id}: {error}");
        }

        // Wait before retrying
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}
