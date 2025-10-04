use std::collections::VecDeque;
use std::time::Instant;
use tracing::debug;
use webrtc_vad::{SampleRate, Vad, VadMode};

use crate::config::AppConfig;
use crate::shared::audio::types::speech_segment::SpeechSegment;
use crate::shared::audio::utils::convert_pcm_bytes_to_wav::convert_pcm_bytes_to_wav;

#[derive(Debug, Clone)]
pub struct SpeechState {
    pub is_in_speech: bool,
    pub consecutive_speech_count: usize,
    pub consecutive_silence_count: usize,
    pub speech_start_frame_idx: usize,
    pub current_speech_samples: Vec<i16>,
    pub total_frame_count: usize,
    pub last_activity: Instant,
}

#[derive(Debug)]
pub struct WebRtcVadProcessResult {
    pub speech_state: SpeechState,
    pub frame_buffer: VecDeque<Vec<i16>>,
    pub segments: Vec<SpeechSegment>,
}

pub struct WebRtcVadProcessor;

impl SpeechState {
    pub fn new() -> Self {
        Self {
            is_in_speech: false,
            consecutive_speech_count: 0,
            consecutive_silence_count: 0,
            speech_start_frame_idx: 0,
            current_speech_samples: Vec::new(),
            total_frame_count: 0,
            last_activity: Instant::now(),
        }
    }
}

impl WebRtcVadProcessor {
    /// Create initial state for new stream
    pub fn create_initial_state() -> (SpeechState, VecDeque<Vec<i16>>) {
        (SpeechState::new(), VecDeque::new())
    }

    /// Process a single frame and return new state with completed speech segments
    pub fn process_frame(
        app_config: &AppConfig,
        speech_state: SpeechState,
        frame_buffer: VecDeque<Vec<i16>>,
        frame: &[i16],
    ) -> WebRtcVadProcessResult {
        let mut segments = Vec::new();
        let mut speech_state = speech_state.clone();

        // Ensure frame is the right size
        let normalized_frame = Self::normalize_frame(app_config, frame);

        // Maintain rolling buffer for padding
        let frame_buffer = Self::update_frame_buffer(app_config, frame_buffer, &normalized_frame);

        // Perform voice activity detection
        let is_voice = Self::detect_voice_activity(app_config, &normalized_frame);

        speech_state.total_frame_count += 1;

        // Update speech/silence counters
        Self::update_speech_counters(&mut speech_state, is_voice);

        // Handle speech state transitions
        if let Some(segment) = Self::handle_speech_transitions(
            app_config,
            &mut speech_state,
            &frame_buffer,
            is_voice,
            &normalized_frame,
        ) {
            segments.push(segment);
        }

        // Add current frame to ongoing speech if in speech mode
        if speech_state.is_in_speech {
            speech_state
                .current_speech_samples
                .extend_from_slice(&normalized_frame);
            speech_state.last_activity = Instant::now();
        }

        WebRtcVadProcessResult {
            speech_state,
            frame_buffer,
            segments,
        }
    }

    /// Normalize frame to expected size
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

    /// Update the rolling frame buffer for padding
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

    /// Detect voice activity in the current frame
    fn detect_voice_activity(app_config: &AppConfig, frame: &[i16]) -> bool {
        let vad_mode = match app_config.webrtc_vad_mode.as_str() {
            "Quality" => VadMode::Quality,
            "LowBitrate" => VadMode::LowBitrate,
            "Aggressive" => VadMode::Aggressive,
            "VeryAggressive" => VadMode::VeryAggressive,
            _ => VadMode::Aggressive,
        };
        let mut webrtc_vad = Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, vad_mode);
        webrtc_vad
            .is_voice_segment(frame)
            .expect("WebRTC VAD should process frame successfully")
    }

    /// Update consecutive speech and silence counters
    fn update_speech_counters(speech_state: &mut SpeechState, is_voice: bool) {
        if is_voice {
            speech_state.consecutive_speech_count += 1;
            speech_state.consecutive_silence_count = 0;
        } else {
            speech_state.consecutive_silence_count += 1;
            speech_state.consecutive_speech_count = 0;
        }
    }

    /// Handle speech state transitions and return completed segments
    fn handle_speech_transitions(
        app_config: &AppConfig,
        speech_state: &mut SpeechState,
        frame_buffer: &VecDeque<Vec<i16>>,
        is_voice: bool,
        _current_frame: &[i16],
    ) -> Option<SpeechSegment> {
        // Start speech detection with debouncing
        if !speech_state.is_in_speech
            && is_voice
            && speech_state.consecutive_speech_count >= app_config.webrtc_vad_debounce_frame_number
        {
            Self::start_speech_segment(app_config, speech_state, frame_buffer);
            return None;
        }

        // End speech detection
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

    /// Start a new speech segment
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

        // Add padding from previous frames
        Self::add_padding_to_speech(app_config, speech_state, frame_buffer);

        debug!(
            "WebRTC VAD: speech started at frame {}",
            speech_state.total_frame_count
        );
    }

    /// Add padding frames to the beginning of speech segment
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

    /// End the current speech segment and return it if valid
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
            let segment =
                Self::create_speech_segment(app_config, speech_state, speech_duration_frames);
            debug!(
                "WebRTC VAD segment completed: {:.2}s - {:.2}s ({:.2}s duration, {} samples)",
                segment.start,
                segment.end,
                segment.end - segment.start,
                speech_state.current_speech_samples.len()
            );

            Self::reset_speech_state(speech_state);
            Some(segment)
        } else {
            debug!(
                "Speech segment too short: {} frames (min: {}), discarding",
                speech_duration_frames, min_speech_frames
            );
            Self::reset_speech_state(speech_state);
            None
        }
    }

    /// Create a speech segment from current state
    fn create_speech_segment(
        app_config: &AppConfig,
        speech_state: &SpeechState,
        _duration_frames: usize,
    ) -> SpeechSegment {
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

        let wav_data = convert_pcm_bytes_to_wav(&pcm_bytes);

        SpeechSegment {
            start: start_time,
            end: end_time,
            audio_data: wav_data,
        }
    }

    /// Reset speech-related state
    fn reset_speech_state(speech_state: &mut SpeechState) {
        speech_state.is_in_speech = false;
        speech_state.consecutive_silence_count = 0;
        speech_state.current_speech_samples.clear();
    }

    /// Finalize any remaining speech segment (for end of stream)
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

                let wav_data = convert_pcm_bytes_to_wav(&pcm_bytes);

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
        }
        None
    }

    /// Check if we should reset due to inactivity (for streaming)
    pub fn should_reset_due_to_inactivity(
        speech_state: &SpeechState,
        max_inactivity_s: u64,
    ) -> bool {
        speech_state.last_activity.elapsed().as_secs() > max_inactivity_s
    }

    /// Get samples per frame for external frame processing
    pub fn samples_per_frame(app_config: &AppConfig) -> usize {
        (app_config.webrtc_vad_sample_rate_number as u64
            * app_config.webrtc_vad_frame_duration_ms as u64
            / 1000) as usize
    }
}
