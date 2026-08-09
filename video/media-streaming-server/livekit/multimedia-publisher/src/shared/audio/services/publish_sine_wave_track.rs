use std::f32::consts::TAU;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use livekit::options::TrackPublishOptions;
use livekit::prelude::*;
use livekit::webrtc::audio_source::native::NativeAudioSource;
use livekit::webrtc::prelude::{AudioFrame, AudioSourceOptions, RtcAudioSource};
use tracing::{error, info};

const SAMPLE_RATE_HZ: u32 = 48_000;
const FRAME_DURATION_MS: u64 = 10;
const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE_HZ as u64 * FRAME_DURATION_MS / 1_000) as usize;
const TONE_FREQUENCY_HZ: f32 = 440.0;
const BEEP_FREQUENCY_HZ: f32 = 1_000.0;
const BEEP_DURATION_MS: u64 = 100;
const TONE_AMPLITUDE: f32 = 0.05;
const BEEP_AMPLITUDE: f32 = 0.3;

pub async fn publish_sine_wave_track(room: &Arc<Room>) -> Result<TrackSid> {
    let source = NativeAudioSource::new(AudioSourceOptions::default(), SAMPLE_RATE_HZ, 1, 0);

    let track =
        LocalAudioTrack::create_audio_track("sine-wave", RtcAudioSource::Native(source.clone()));

    let publication = room
        .local_participant()
        .publish_track(
            LocalTrack::Audio(track),
            TrackPublishOptions {
                source: TrackSource::Microphone,
                ..Default::default()
            },
        )
        .await
        .map_err(|error| anyhow!("Failed to publish audio track: {error}"))?;
    let track_sid = publication.sid();

    info!("Published audio track {track_sid}");

    tokio::spawn(run_sine_wave_capture_loop(source));

    Ok(track_sid)
}

async fn run_sine_wave_capture_loop(source: NativeAudioSource) {
    let mut ticker = tokio::time::interval(Duration::from_millis(FRAME_DURATION_MS));
    let mut sample_index: u64 = 0;

    loop {
        ticker.tick().await;

        let mut data = vec![0i16; SAMPLES_PER_FRAME];
        for sample in data.iter_mut() {
            let elapsed_ms = sample_index * 1_000 / SAMPLE_RATE_HZ as u64;
            let is_beeping = elapsed_ms % 1_000 < BEEP_DURATION_MS;
            let (frequency, amplitude) = if is_beeping {
                (BEEP_FREQUENCY_HZ, BEEP_AMPLITUDE)
            } else {
                (TONE_FREQUENCY_HZ, TONE_AMPLITUDE)
            };

            let phase = TAU * frequency * (sample_index as f32 / SAMPLE_RATE_HZ as f32);
            *sample = (phase.sin() * amplitude * i16::MAX as f32) as i16;
            sample_index += 1;
        }

        let frame = AudioFrame {
            data: data.into(),
            samples_per_channel: SAMPLES_PER_FRAME as u32,
            sample_rate: SAMPLE_RATE_HZ,
            num_channels: 1,
        };

        if let Err(error) = source.capture_frame(&frame).await {
            error!("Failed to capture audio frame: {error}");
        }
    }
}
