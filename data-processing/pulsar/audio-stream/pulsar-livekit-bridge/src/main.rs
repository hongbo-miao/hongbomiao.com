#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod shared;

use crate::config::AppConfig;
use crate::shared::audio::services::bridge_pulsar_to_livekit::bridge_pulsar_to_livekit;
use anyhow::{Result, anyhow};
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::{Json, Router};
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::get();

    let token_state = TokenServerState {
        livekit_api_key: Arc::new(config.livekit_api_key.clone()),
        livekit_api_secret: Arc::new(config.livekit_api_secret.clone()),
        livekit_room: Arc::new(config.livekit_room.clone()),
    };

    let app = Router::new()
        .route("/token", axum::routing::get(get_token))
        .with_state(token_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.http_port));
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|error| anyhow!("Failed to bind HTTP listener: {error}"))?;

    info!("Token server listening on {addr}");

    let bridge_future = bridge_pulsar_to_livekit(
        &config.pulsar_url,
        &config.pulsar_topic,
        &config.livekit_url,
        &config.livekit_api_key,
        &config.livekit_api_secret,
        &config.livekit_room,
        &config.hostname,
    );

    tokio::select! {
        result = axum::serve(listener, app) => {
            result.map_err(|error| anyhow!("HTTP server error: {error}"))?;
        }
        result = bridge_future => {
            result?;
        }
    }

    Ok(())
}
