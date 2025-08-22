use crate::shared::openai::types::chat_response::ChatResponse;
use crate::shared::openai::utils::chat::chat;
use async_graphql::{Object, SimpleObject};

#[derive(SimpleObject)]
pub struct HelloResponse {
    pub message: String,
}

pub struct Query;

#[Object]
impl Query {
    async fn hello(&self) -> HelloResponse {
        HelloResponse {
            message: "Hello, world!".to_string(),
        }
    }

    async fn chat(&self, message: String) -> Result<ChatResponse, String> {
        chat(message).await
    }
}

#[cfg(test)]
#[path = "query_test.rs"]
mod tests;
