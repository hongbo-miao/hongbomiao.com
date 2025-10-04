use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSentEventMessage {
    #[serde(rename = "type")]
    pub message_type: String,
    pub stream_id: String,
    pub transcription: String,
    pub timestamp: f64,
}
