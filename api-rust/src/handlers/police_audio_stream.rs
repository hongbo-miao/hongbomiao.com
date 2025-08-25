use std::sync::Arc;

use axum::{
    extract::{Query, ws::WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
};
use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::shared::police_audio_stream::constants::police_streams::POLICE_STREAMS;
use crate::shared::police_audio_stream::utils::handle_police_audio_websocket::handle_police_audio_websocket;
use crate::shared::police_audio_stream::utils::police_audio_stream_manager::{
    PoliceAudioStreamManager, PoliceAudioStreamState,
};

pub static POLICE_STREAM_STATE: Lazy<Arc<PoliceAudioStreamState>> =
    Lazy::new(|| Arc::new(PoliceAudioStreamState::new()));

#[derive(Deserialize)]
pub struct WebSocketParams {
    pub police_stream_id: String,
}

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
