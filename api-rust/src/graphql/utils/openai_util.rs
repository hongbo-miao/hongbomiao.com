use async_graphql::SimpleObject;
use openai_api_rs::v1::api::OpenAIClient;
use openai_api_rs::v1::chat_completion::{self, ChatCompletionRequest};
use std::env;

#[derive(SimpleObject)]
pub struct ChatResponse {
    pub content: String,
}

pub async fn chat(message: String) -> Result<ChatResponse, String> {
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY not set in environment".to_string())?;

    let base_url = env::var("OPENAI_BASE_URL")
        .map_err(|_| "OPENAI_BASE_URL not set in environment".to_string())?;

    let model =
        env::var("OPENAI_MODEL").map_err(|_| "OPENAI_MODEL not set in environment".to_string())?;

    let mut client = OpenAIClient::builder()
        .with_api_key(api_key)
        .with_endpoint(&base_url)
        .build()
        .map_err(|e| format!("Failed to create OpenAI client: {}", e))?;

    let req = ChatCompletionRequest::new(
        model,
        vec![chat_completion::ChatCompletionMessage {
            role: chat_completion::MessageRole::user,
            content: chat_completion::Content::Text(message),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
    );

    let result = client
        .chat_completion(req)
        .await
        .map_err(|e| format!("OpenAI API error: {}", e))?;

    let content = result.choices[0]
        .message
        .content
        .as_ref()
        .ok_or_else(|| "No content in response".to_string())?;

    Ok(ChatResponse {
        content: content.clone(),
    })
}
