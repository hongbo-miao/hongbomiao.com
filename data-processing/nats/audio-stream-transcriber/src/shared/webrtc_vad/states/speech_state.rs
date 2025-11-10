use std::time::Instant;

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
