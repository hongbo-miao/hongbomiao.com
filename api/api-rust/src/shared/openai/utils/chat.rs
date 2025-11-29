use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use async_openai::{Client, config::OpenAIConfig};

use crate::config::AppConfig;
use crate::shared::openai::types::chat_response::ChatResponse;

pub async fn chat(message: String) -> Result<ChatResponse, String> {
    let config = AppConfig::get();

    let openai_config = OpenAIConfig::new()
        .with_api_key(&config.openai_api_key)
        .with_api_base(format!("{}/v1", config.openai_api_base_url));

    let client = Client::with_config(openai_config);

    let request = CreateChatCompletionRequestArgs::default()
        .model(&config.openai_model)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content(message)
            .build()
            .map_err(|error| format!("Failed to build message: {}", error))?
            .into()])
        .build()
        .map_err(|error| format!("Failed to build request: {}", error))?;

    let result = client
        .chat()
        .create(request)
        .await
        .map_err(|error| format!("OpenAI API error: {}", error))?;

    let content = result.choices[0]
        .message
        .content
        .as_ref()
        .ok_or_else(|| "No content in response".to_string())?;

    Ok(ChatResponse {
        content: content.clone(),
    })
}
