use crate::config::AppConfig;
use crate::shared::audio::utils::convert_pcm_bytes_to_wav::convert_pcm_bytes_to_wav;
use crate::shared::webrtc_vad::states::speech_state::SpeechState;
use crate::shared::webrtc_vad::types::speech_segment::SpeechSegment;
use crate::shared::webrtc_vad::types::webrtc_vad_process_result::WebRtcVadProcessResult;
use std::collections::VecDeque;
use tracing::{debug, error, info};
use webrtc_vad::Vad;

pub struct WebRtcVadProcessor;

impl WebRtcVadProcessor {
    pub fn create_initial_state() -> (SpeechState, VecDeque<Vec<i16>>) {
        (SpeechState::new(), VecDeque::new())
    }

    pub fn process_frame(
        app_config: &AppConfig,
        webrtc_vad: &mut Vad,
        mut speech_state: SpeechState,
        frame_buffer: VecDeque<Vec<i16>>,
        frame: &[i16],
    ) -> WebRtcVadProcessResult {
        let mut segments = Vec::new();

        let normalized_frame = Self::normalize_frame(app_config, frame);
        let frame_buffer = Self::update_frame_buffer(app_config, frame_buffer, &normalized_frame);
        let is_voice = Self::detect_voice_activity(webrtc_vad, &normalized_frame);

        speech_state.total_frame_count += 1;

        Self::update_speech_counters(&mut speech_state, is_voice);

        if let Some(segment) = Self::handle_speech_transitions(
            app_config,
            &mut speech_state,
            &frame_buffer,
            is_voice,
            &normalized_frame,
        ) {
            segments.push(segment);
        }

        if speech_state.is_in_speech {
            speech_state
                .current_speech_samples
                .extend_from_slice(&normalized_frame);
            speech_state.last_activity = std::time::Instant::now();
        }

        WebRtcVadProcessResult {
            speech_state,
            frame_buffer,
            segments,
        }
    }

    fn normalize_frame(app_config: &AppConfig, frame: &[i16]) -> Vec<i16> {
        let samples_per_frame = (app_config.webrtc_vad_sample_rate_number as u64
            * app_config.webrtc_vad_frame_duration_ms as u64
            / 1000) as usize;
        let mut normalized = frame.to_vec();
        if normalized.len() < samples_per_frame {
            normalized.resize(samples_per_frame, 0);
        } else if normalized.len() > samples_per_frame {
            normalized.truncate(samples_per_frame);
        }
        normalized
    }

    fn update_frame_buffer(
        app_config: &AppConfig,
        mut frame_buffer: VecDeque<Vec<i16>>,
        frame: &[i16],
    ) -> VecDeque<Vec<i16>> {
        frame_buffer.push_back(frame.to_vec());
        let padding_frames = (app_config
            .webrtc_vad_padding_duration_ms
            .div_ceil(app_config.webrtc_vad_frame_duration_ms)
            as usize)
            .max(1);
        let max_buffer_size = padding_frames.max(10);
        while frame_buffer.len() > max_buffer_size {
            frame_buffer.pop_front();
        }
        frame_buffer
    }

    fn detect_voice_activity(webrtc_vad: &mut Vad, frame: &[i16]) -> bool {
        webrtc_vad
            .is_voice_segment(frame)
            .expect("WebRTC VAD should process frame successfully")
    }

    fn update_speech_counters(speech_state: &mut SpeechState, is_voice: bool) {
        if is_voice {
            speech_state.consecutive_speech_count += 1;
            speech_state.consecutive_silence_count = 0;
        } else {
            speech_state.consecutive_silence_count += 1;
            speech_state.consecutive_speech_count = 0;
        }
    }

    fn handle_speech_transitions(
        app_config: &AppConfig,
        speech_state: &mut SpeechState,
        frame_buffer: &VecDeque<Vec<i16>>,
        is_voice: bool,
        _current_frame: &[i16],
    ) -> Option<SpeechSegment> {
        if !speech_state.is_in_speech
            && is_voice
            && speech_state.consecutive_speech_count >= app_config.webrtc_vad_debounce_frame_number
        {
            Self::start_speech_segment(app_config, speech_state, frame_buffer);
            return None;
        }

        if speech_state.is_in_speech {
            let silence_frames_to_end = (app_config
                .webrtc_vad_min_silence_duration_ms
                .div_ceil(app_config.webrtc_vad_frame_duration_ms)
                as usize)
                .max(1);
            let padding_frames = (app_config
                .webrtc_vad_padding_duration_ms
                .div_ceil(app_config.webrtc_vad_frame_duration_ms)
                as usize)
                .max(1);
            let end_needed = silence_frames_to_end.max(padding_frames);
            if speech_state.consecutive_silence_count >= end_needed {
                return Self::end_speech_segment(app_config, speech_state);
            }
        }

        None
    }

    fn start_speech_segment(
        app_config: &AppConfig,
        speech_state: &mut SpeechState,
        frame_buffer: &VecDeque<Vec<i16>>,
    ) {
        speech_state.is_in_speech = true;
        speech_state.speech_start_frame_idx = speech_state
            .total_frame_count
            .saturating_sub(speech_state.consecutive_speech_count);
        speech_state.current_speech_samples.clear();

        Self::add_padding_to_speech(app_config, speech_state, frame_buffer);

        info!(
            "WebRTC VAD: speech started at frame {}",
            speech_state.total_frame_count
        );
    }

    fn add_padding_to_speech(
        app_config: &AppConfig,
        speech_state: &mut SpeechState,
        frame_buffer: &VecDeque<Vec<i16>>,
    ) {
        let samples_per_frame = (app_config.webrtc_vad_sample_rate_number as u64
            * app_config.webrtc_vad_frame_duration_ms as u64
            / 1000) as usize;
        let padding_frames = (app_config
            .webrtc_vad_padding_duration_ms
            .div_ceil(app_config.webrtc_vad_frame_duration_ms)
            as usize)
            .max(1);
        let padding_sample_count = padding_frames.min(frame_buffer.len()) * samples_per_frame;
        if padding_sample_count > 0 {
            let start_idx = frame_buffer
                .len()
                .saturating_sub(padding_sample_count / samples_per_frame);
            for frame in frame_buffer.iter().skip(start_idx) {
                speech_state.current_speech_samples.extend_from_slice(frame);
            }
        }
    }

    fn end_speech_segment(
        app_config: &AppConfig,
        speech_state: &mut SpeechState,
    ) -> Option<SpeechSegment> {
        let speech_duration_frames = speech_state.total_frame_count
            - speech_state.speech_start_frame_idx
            - speech_state.consecutive_silence_count;

        let min_speech_frames = (app_config
            .webrtc_vad_min_speech_duration_ms
            .div_ceil(app_config.webrtc_vad_frame_duration_ms)
            as usize)
            .max(1);
        let samples_per_frame = (app_config.webrtc_vad_sample_rate_number as u64
            * app_config.webrtc_vad_frame_duration_ms as u64
            / 1000) as usize;

        if speech_duration_frames >= min_speech_frames
            && speech_state.current_speech_samples.len() >= min_speech_frames * samples_per_frame
        {
            if let Some(segment) =
                Self::create_speech_segment(app_config, speech_state, speech_duration_frames)
            {
                info!(
                    "WebRTC VAD segment completed: {:.2}s - {:.2}s ({:.2}s duration, {} samples)",
                    segment.start,
                    segment.end,
                    segment.end - segment.start,
                    speech_state.current_speech_samples.len()
                );

                Self::reset_speech_state(speech_state);
                Some(segment)
            } else {
                info!("Failed to create WAV for speech segment, discarding segment");
                Self::reset_speech_state(speech_state);
                None
            }
        } else {
            info!(
                "Speech segment too short: {speech_duration_frames} frames (min: {min_speech_frames}), discarding",
            );
            Self::reset_speech_state(speech_state);
            None
        }
    }

    fn create_speech_segment(
        app_config: &AppConfig,
        speech_state: &SpeechState,
        _duration_frames: usize,
    ) -> Option<SpeechSegment> {
        let start_time = (speech_state.speech_start_frame_idx as f64
            * app_config.webrtc_vad_frame_duration_ms as f64)
            / 1000.0;
        let end_time = ((speech_state.total_frame_count - speech_state.consecutive_silence_count)
            as f64
            * app_config.webrtc_vad_frame_duration_ms as f64)
            / 1000.0;

        let pcm_bytes: Vec<u8> = speech_state
            .current_speech_samples
            .iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();

        let wav_data = match convert_pcm_bytes_to_wav(&pcm_bytes) {
            Ok(data) => data,
            Err(error) => {
                error!(
                    "Failed to convert PCM to WAV for segment ({:.2}s - {:.2}s): {}",
                    start_time, end_time, error
                );
                return None;
            }
        };

        Some(SpeechSegment {
            start: start_time,
            end: end_time,
            audio_data: wav_data,
        })
    }

    fn reset_speech_state(speech_state: &mut SpeechState) {
        speech_state.is_in_speech = false;
        speech_state.consecutive_silence_count = 0;
        speech_state.current_speech_samples.clear();
    }

    pub fn finalize(app_config: &AppConfig, speech_state: SpeechState) -> Option<SpeechSegment> {
        if speech_state.is_in_speech && !speech_state.current_speech_samples.is_empty() {
            let speech_duration_frames =
                speech_state.total_frame_count - speech_state.speech_start_frame_idx;
            let min_speech_frames = (app_config
                .webrtc_vad_min_speech_duration_ms
                .div_ceil(app_config.webrtc_vad_frame_duration_ms)
                as usize)
                .max(1);
            if speech_duration_frames >= min_speech_frames {
                let start_time = (speech_state.speech_start_frame_idx as f64
                    * app_config.webrtc_vad_frame_duration_ms as f64)
                    / 1000.0;
                let end_time = (speech_state.total_frame_count as f64
                    * app_config.webrtc_vad_frame_duration_ms as f64)
                    / 1000.0;

                let pcm_bytes: Vec<u8> = speech_state
                    .current_speech_samples
                    .iter()
                    .flat_map(|&sample| sample.to_le_bytes())
                    .collect();

                match convert_pcm_bytes_to_wav(&pcm_bytes) {
                    Ok(wav_data) => {
                        debug!(
                            "WebRTC VAD final segment: {:.2}s - {:.2}s ({:.2}s duration, {} samples)",
                            start_time,
                            end_time,
                            end_time - start_time,
                            speech_state.current_speech_samples.len()
                        );

                        return Some(SpeechSegment {
                            start: start_time,
                            end: end_time,
                            audio_data: wav_data,
                        });
                    }
                    Err(error) => {
                        error!(
                            "Failed to convert PCM to WAV for final segment ({:.2}s - {:.2}s): {}",
                            start_time, end_time, error
                        );
                        return None;
                    }
                }
            }
        }
        None
    }

    pub fn should_reset_due_to_inactivity(
        speech_state: &SpeechState,
        max_inactivity_s: u64,
    ) -> bool {
        speech_state.last_activity.elapsed().as_secs() > max_inactivity_s
    }

    pub fn samples_per_frame(app_config: &AppConfig) -> usize {
        (app_config.webrtc_vad_sample_rate_number as u64
            * app_config.webrtc_vad_frame_duration_ms as u64
            / 1000) as usize
    }
}
