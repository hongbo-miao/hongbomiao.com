use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub stream_id: String,
    pub timestamp_ns: i64,
    pub text: String,
    pub language: String,
    pub duration_s: f64,
    pub segment_start_s: f64,
    pub segment_end_s: f64,
    pub words: Vec<TranscriptWord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptWord {
    pub word: String,
    pub start_s: f64,
    pub end_s: f64,
    pub probability: f64,
}
