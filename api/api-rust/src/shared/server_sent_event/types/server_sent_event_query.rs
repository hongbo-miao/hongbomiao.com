use serde::Deserialize;
use utoipa::ToSchema;

#[derive(Debug, Deserialize, ToSchema)]
pub struct ServerSentEventQuery {
    pub message_type: String,
    #[serde(default)]
    pub stream: bool,
}
