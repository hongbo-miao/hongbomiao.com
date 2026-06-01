//! `GET /audio/{channel_id}` - upgrades to a WebSocket and streams that
//! channel's post-DSP audio as little-endian i16 PCM binary frames. The browser
//! plays the PCM through an AudioWorklet. No codec, no transcoding.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, warn};

use crate::channel::ChannelEndpoint;
use crate::web::app_state::AppState;

pub async fn serve_channel_audio(
    upgrade: WebSocketUpgrade,
    Path(channel_id): Path<usize>,
    State(state): State<AppState>,
) -> Response {
    let Some(endpoint) = state.channels.get(channel_id).cloned() else {
        return (StatusCode::NOT_FOUND, "no such channel").into_response();
    };
    upgrade.on_upgrade(move |socket| forward_audio(socket, endpoint))
}

/// Subscribe to the channel's audio broadcast and forward each frame as PCM
/// until the client disconnects or the channel closes.
async fn forward_audio(mut socket: WebSocket, endpoint: ChannelEndpoint) {
    let mut audio_receiver = endpoint.audio_sender.subscribe();
    let channel_name = &endpoint.name;
    debug!("Listener connected to {channel_name}");

    loop {
        match audio_receiver.recv().await {
            Ok(frame) => {
                let mut bytes = Vec::with_capacity(frame.len() * 2);
                for sample in frame.iter() {
                    let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    bytes.extend_from_slice(&scaled.to_le_bytes());
                }
                if socket.send(Message::Binary(bytes.into())).await.is_err() {
                    break;
                }
            }
            Err(RecvError::Lagged(count)) => {
                warn!("Listener on {channel_name} lagged, skipped {count} frames");
            }
            Err(RecvError::Closed) => break,
        }
    }

    debug!("Listener disconnected from {channel_name}");
}
