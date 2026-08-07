use serde::Serialize;

use crate::agent::types::chat_completion_choice::ChatCompletionChoice;

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}
