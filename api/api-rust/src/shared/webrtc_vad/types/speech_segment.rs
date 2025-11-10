#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start: f64,
    pub end: f64,
    pub audio_data: Vec<u8>,
}
