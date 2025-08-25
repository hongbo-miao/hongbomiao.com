use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ServerSentEventQuery {
    pub message_type: String,
    #[serde(default)]
    pub stream: bool,
}
