use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub duration: f64,
    #[serde(default)]
    pub words: Vec<TranscriptionWord>,
    #[serde(default)]
    pub segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub id: i32,
    pub seek: i32,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<i32>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
}
