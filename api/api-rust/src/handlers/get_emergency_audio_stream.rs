use axum::{
    extract::{Query, ws::WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use utoipa::ToSchema;

use crate::shared::emergency_audio_stream::constants::emergency_streams::EMERGENCY_STREAMS;
use crate::shared::emergency_audio_stream::utils::emergency_audio_stream_manager::FireAudioStreamManager;
use crate::shared::emergency_audio_stream::utils::emergency_stream_state::EMERGENCY_STREAM_STATE;
use crate::shared::emergency_audio_stream::utils::handle_emergency_audio_websocket::handle_emergency_audio_websocket;

#[derive(Deserialize, ToSchema, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct WebSocketParams {
    /// ID of the fire stream to connect to
    pub emergency_stream_id: String,
}

/// WebSocket endpoint for fire audio streaming
#[utoipa::path(
    get,
    path = "/ws/emergency-audio-stream",
    params(
        ("emergency_stream_id" = String, Query, description = "ID of the fire stream to connect to")
    ),
    responses(
        (status = 101, description = "WebSocket connection established"),
        (status = 400, description = "Unknown emergency_stream_id", body = String)
    ),
    tag = "streaming"
)]
pub async fn get_emergency_audio_stream(
    web_socket_upgrade: WebSocketUpgrade,
    Query(params): Query<WebSocketParams>,
) -> impl IntoResponse {
    if !EMERGENCY_STREAMS.contains_key(params.emergency_stream_id.as_str()) {
        return (StatusCode::BAD_REQUEST, "Unknown emergency_stream_id").into_response();
    }

    FireAudioStreamManager::start_stream(&EMERGENCY_STREAM_STATE, &params.emergency_stream_id)
        .await;
    web_socket_upgrade.on_upgrade(move |web_socket| {
        handle_emergency_audio_websocket(
            web_socket,
            EMERGENCY_STREAM_STATE.clone(),
            params.emergency_stream_id,
        )
    })
}
