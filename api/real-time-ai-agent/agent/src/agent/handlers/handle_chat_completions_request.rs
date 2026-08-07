use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::{StreamExt, stream};
use rig::agent::{Agent, MultiTurnStreamItem};
use rig::completion::Prompt;
use rig::prelude::StreamingPrompt;
use rig::providers::openrouter::completion::CompletionModel;
use rig::streaming::StreamedAssistantContent;
use serde_json::json;

use crate::agent::types::chat_completion_choice::ChatCompletionChoice;
use crate::agent::types::chat_completion_request::ChatCompletionRequest;
use crate::agent::types::chat_completion_response::ChatCompletionResponse;
use crate::agent::types::chat_message::ChatMessage;

fn current_unix_timestamp_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_secs() as i64
}

/// POST /v1/chat/completions
pub async fn handle_chat_completions_request(
    State(agent): State<Arc<Agent<CompletionModel>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let latest_user_question = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.clone())
        .unwrap_or_default();

    let completion_id = format!("chatcmpl-{}", current_unix_timestamp_seconds());
    let created = current_unix_timestamp_seconds();

    if request.stream {
        let turn_stream = agent
            .stream_prompt(latest_user_question)
            .max_turns(10)
            .await;

        let model_id = request.model.clone();
        let event_stream = turn_stream.flat_map(move |item| {
            let events: Vec<Result<Event, Infallible>> = match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => vec![Ok(Event::default().data(
                    json!({
                        "id": completion_id.clone(),
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id.clone(),
                        "choices": [{
                            "index": 0,
                            "delta": { "content": text.text },
                            "finish_reason": null,
                        }],
                    })
                    .to_string(),
                ))],
                Ok(MultiTurnStreamItem::FinalResponse(_)) => vec![
                    Ok(Event::default().data(
                        json!({
                            "id": completion_id.clone(),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id.clone(),
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        })
                        .to_string(),
                    )),
                    Ok(Event::default().data("[DONE]")),
                ],
                Err(error) => vec![Ok(Event::default().data(
                    json!({
                        "id": completion_id.clone(),
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id.clone(),
                        "choices": [{
                            "index": 0,
                            "delta": { "content": format!("Agent error: {error}") },
                            "finish_reason": "stop",
                        }],
                    })
                    .to_string(),
                ))],
                // Tool-call deltas, tool results, reasoning, and per-turn
                // completion-call bookkeeping aren't shown as answer text.
                Ok(_) => vec![],
            };
            stream::iter(events)
        });

        Sse::new(event_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let answer = match agent.prompt(latest_user_question).max_turns(10).await {
            Ok(answer) => answer,
            Err(error) => format!("Agent error: {error}"),
        };

        Json(ChatCompletionResponse {
            id: completion_id,
            object: "chat.completion".to_string(),
            created,
            model: request.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: answer,
                },
                finish_reason: "stop".to_string(),
            }],
        })
        .into_response()
    }
}
