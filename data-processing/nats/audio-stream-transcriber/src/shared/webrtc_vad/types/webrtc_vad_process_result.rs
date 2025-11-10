use crate::shared::webrtc_vad::states::speech_state::SpeechState;
use crate::shared::webrtc_vad::types::speech_segment::SpeechSegment;
use std::collections::VecDeque;

#[derive(Debug)]
pub struct WebRtcVadProcessResult {
    pub speech_state: SpeechState,
    pub frame_buffer: VecDeque<Vec<i16>>,
    pub segments: Vec<SpeechSegment>,
}
