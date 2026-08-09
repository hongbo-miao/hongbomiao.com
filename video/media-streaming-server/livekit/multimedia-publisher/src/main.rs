#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod shared;

use crate::config::AppConfig;
use crate::shared::audio::services::publish_sine_wave_track::publish_sine_wave_track;
use crate::shared::egress::services::start_track_composite_hls_egress::start_track_composite_hls_egress;
use crate::shared::egress::services::stop_hls_egress::stop_hls_egress;
use crate::shared::process::utils::wait_for_shutdown_signal::wait_for_shutdown_signal;
use crate::shared::video::services::publish_stepping_bar_track::publish_stepping_bar_track;
use anyhow::{Result, anyhow};
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::{Json, Router};
use livekit::prelude::*;
use livekit_api::access_token::{AccessToken, VideoGrants};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

#[derive(Clone)]
struct TokenServerState {
    livekit_api_key: Arc<String>,
    livekit_api_secret: Arc<String>,
    livekit_room: Arc<String>,
    hls_playlist_url: Arc<String>,
}

#[derive(Deserialize)]
struct TokenQuery {
    identity: Option<String>,
}

#[derive(Serialize)]
struct TokenResponse {
    token: String,
    livekit_url: String,
}

#[derive(Serialize)]
struct StreamResponse {
    playlist_url: String,
}

async fn get_token(
    Query(query): Query<TokenQuery>,
    State(state): State<TokenServerState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let host = headers
        .get("host")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("localhost");
    let livekit_url = format!(
        "ws://{}:7880",
        host.split(':').next().unwrap_or("localhost")
    );

    let identity = query.identity.unwrap_or_else(|| {
        let uuid = uuid::Uuid::new_v4().to_string();
        format!("viewer-{}", &uuid[..8])
    });

    let result = AccessToken::with_api_key(&state.livekit_api_key, &state.livekit_api_secret)
        .with_identity(&identity)
        .with_grants(VideoGrants {
            room_join: true,
            room: state.livekit_room.as_ref().clone(),
            can_publish: false,
            can_subscribe: true,
            ..Default::default()
        })
        .to_jwt();

    match result {
        Ok(token) => Json(TokenResponse { token, livekit_url }).into_response(),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to generate token: {error}"),
        )
            .into_response(),
    }
}

async fn get_stream(State(state): State<TokenServerState>) -> impl IntoResponse {
    Json(StreamResponse {
        playlist_url: state.hls_playlist_url.as_ref().clone(),
    })
}

fn generate_publisher_token(config: &AppConfig) -> Result<String> {
    AccessToken::with_api_key(&config.livekit_api_key, &config.livekit_api_secret)
        .with_identity("multimedia-publisher")
        .with_grants(VideoGrants {
            room_join: true,
            room: config.livekit_room.clone(),
            can_publish: true,
            can_subscribe: false,
            ..Default::default()
        })
        .to_jwt()
        .map_err(|error| anyhow!("Failed to generate publisher token: {error}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::get();

    let publisher_token = generate_publisher_token(config)?;

    info!(
        "Connecting to LiveKit at {}, room {}",
        config.livekit_url, config.livekit_room
    );
    let (room, _room_event_receiver) = Room::connect(
        &config.livekit_url,
        &publisher_token,
        RoomOptions::default(),
    )
    .await
    .map_err(|error| anyhow!("Failed to connect to LiveKit room: {error}"))?;
    let room = Arc::new(room);

    info!("Connected to LiveKit room {}", config.livekit_room);

    let audio_track_sid = publish_sine_wave_track(&room).await?;
    let video_track_sid = publish_stepping_bar_track(&room).await?;

    let hls_egress_session =
        start_track_composite_hls_egress(config, &audio_track_sid, &video_track_sid).await?;

    let token_state = TokenServerState {
        livekit_api_key: Arc::new(config.livekit_api_key.clone()),
        livekit_api_secret: Arc::new(config.livekit_api_secret.clone()),
        livekit_room: Arc::new(config.livekit_room.clone()),
        hls_playlist_url: Arc::new(format!("/hls/{}", hls_egress_session.live_playlist_key)),
    };

    let app = Router::new()
        .route("/token", axum::routing::get(get_token))
        .route("/stream", axum::routing::get(get_stream))
        .with_state(token_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.http_port));
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|error| anyhow!("Failed to bind HTTP listener: {error}"))?;

    info!("Token server listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(wait_for_shutdown_signal())
        .await
        .map_err(|error| anyhow!("HTTP server error: {error}"))?;

    stop_hls_egress(config, &hls_egress_session.egress_id).await?;

    Ok(())
}
