use crate::shared::openai::types::chat_response::ChatResponse;
use crate::shared::openai::utils::chat::chat;
use crate::shared::police_audio_stream::constants::police_streams::POLICE_STREAMS;
use crate::shared::police_audio_stream::utils::police_stream_state::POLICE_STREAM_STATE;
use async_graphql::{Object, SimpleObject};
use utoipa::ToSchema;

#[derive(SimpleObject, ToSchema)]
pub struct HelloResponse {
    pub message: String,
}

#[derive(SimpleObject, ToSchema)]
pub struct PoliceStream {
    pub id: String,
    pub name: String,
    pub location: String,
}

#[derive(SimpleObject, ToSchema)]
pub struct PoliceStreamStatus {
    pub id: String,
    pub client_count: i32,
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

    #[graphql(name = "policeStreams")]
    async fn police_streams(&self) -> Vec<PoliceStream> {
        POLICE_STREAMS
            .iter()
            .map(|(id, info)| PoliceStream {
                id: (*id).to_string(),
                name: info.name.to_string(),
                location: info.location.to_string(),
            })
            .collect()
    }

    #[graphql(name = "policeStreamStatus")]
    async fn police_stream_status(&self) -> Vec<PoliceStreamStatus> {
        let active = POLICE_STREAM_STATE.active_clients.read().await;
        active
            .iter()
            .map(|(k, v)| PoliceStreamStatus {
                id: k.clone(),
                client_count: v.len() as i32,
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "query_test.rs"]
mod tests;
