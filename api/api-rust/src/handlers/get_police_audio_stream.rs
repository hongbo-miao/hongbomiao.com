use axum::{
    extract::{Query, ws::WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use utoipa::ToSchema;

use crate::shared::police_audio_stream::constants::police_streams::POLICE_STREAMS;
use crate::shared::police_audio_stream::utils::handle_police_audio_websocket::handle_police_audio_websocket;
use crate::shared::police_audio_stream::utils::police_audio_stream_manager::PoliceAudioStreamManager;
use crate::shared::police_audio_stream::utils::police_stream_state::POLICE_STREAM_STATE;

#[derive(Deserialize, ToSchema, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct WebSocketParams {
    /// ID of the police stream to connect to
    pub police_stream_id: String,
}

/// WebSocket endpoint for police audio streaming
#[utoipa::path(
    get,
    path = "/ws/police-audio-stream",
    params(
        ("police_stream_id" = String, Query, description = "ID of the police stream to connect to")
    ),
    responses(
        (status = 101, description = "WebSocket connection established"),
        (status = 400, description = "Unknown police_stream_id", body = String)
    ),
    tag = "streaming"
)]
pub async fn get_police_audio_stream(
    web_socket_upgrade: WebSocketUpgrade,
    Query(params): Query<WebSocketParams>,
) -> impl IntoResponse {
    if !POLICE_STREAMS.contains_key(params.police_stream_id.as_str()) {
        return (StatusCode::BAD_REQUEST, "Unknown police_stream_id").into_response();
    }

    PoliceAudioStreamManager::start_stream(&POLICE_STREAM_STATE, &params.police_stream_id).await;
    web_socket_upgrade.on_upgrade(move |web_socket| {
        handle_police_audio_websocket(
            web_socket,
            POLICE_STREAM_STATE.clone(),
            params.police_stream_id,
        )
    })
}
