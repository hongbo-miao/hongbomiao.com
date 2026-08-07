use serde::Serialize;

use crate::agent::types::chat_message::ChatMessage;

#[derive(Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}
