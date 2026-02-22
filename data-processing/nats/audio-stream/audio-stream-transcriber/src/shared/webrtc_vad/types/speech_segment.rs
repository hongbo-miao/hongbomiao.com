#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start_s: f64,
    pub end_s: f64,
    pub pcm_samples: Vec<i16>,
}
