use axum::{
    extract::{Query, ws::WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use utoipa::ToSchema;

use crate::shared::fire_audio_stream::constants::fire_streams::FIRE_STREAMS;
use crate::shared::fire_audio_stream::utils::fire_audio_stream_manager::FireAudioStreamManager;
use crate::shared::fire_audio_stream::utils::fire_stream_state::FIRE_STREAM_STATE;
use crate::shared::fire_audio_stream::utils::handle_fire_audio_websocket::handle_fire_audio_websocket;

#[derive(Deserialize, ToSchema, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct WebSocketParams {
    /// ID of the fire stream to connect to
    pub fire_stream_id: String,
}

/// WebSocket endpoint for fire audio streaming
#[utoipa::path(
    get,
    path = "/ws/fire-audio-stream",
    params(
        ("fire_stream_id" = String, Query, description = "ID of the fire stream to connect to")
    ),
    responses(
        (status = 101, description = "WebSocket connection established"),
        (status = 400, description = "Unknown fire_stream_id", body = String)
    ),
    tag = "streaming"
)]
pub async fn get_fire_audio_stream(
    web_socket_upgrade: WebSocketUpgrade,
    Query(params): Query<WebSocketParams>,
) -> impl IntoResponse {
    if !FIRE_STREAMS.contains_key(params.fire_stream_id.as_str()) {
        return (StatusCode::BAD_REQUEST, "Unknown fire_stream_id").into_response();
    }

    FireAudioStreamManager::start_stream(&FIRE_STREAM_STATE, &params.fire_stream_id).await;
    web_socket_upgrade.on_upgrade(move |web_socket| {
        handle_fire_audio_websocket(web_socket, FIRE_STREAM_STATE.clone(), params.fire_stream_id)
    })
}
