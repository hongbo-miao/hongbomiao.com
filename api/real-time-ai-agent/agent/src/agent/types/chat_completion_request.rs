use serde::Deserialize;

use crate::agent::types::chat_message::ChatMessage;

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
}
