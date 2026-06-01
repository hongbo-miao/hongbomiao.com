//! Builds the axum router (channel list + per-channel audio WebSocket + static
//! listener page) and runs it under the shared cancellation token.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use axum::routing::get;
use tokio_util::sync::CancellationToken;
use tower_http::services::ServeDir;
use tracing::{info, warn};

use crate::channel::ChannelEndpoint;
use crate::web::app_state::AppState;
use crate::web::list_channels::list_channels;
use crate::web::serve_channel_audio::serve_channel_audio;

pub async fn start_web_server(
    bind: &str,
    static_dir: PathBuf,
    channels: Vec<ChannelEndpoint>,
    cancel: CancellationToken,
) -> Result<tokio::task::JoinHandle<()>> {
    if !static_dir.is_dir() {
        warn!(
            "Static directory {} not found; the listener page will 404. Run from the project root.",
            static_dir.display()
        );
    }

    let state = AppState {
        channels: Arc::new(channels),
    };

    let router = Router::new()
        .route("/channels", get(list_channels))
        .route("/audio/{channel_id}", get(serve_channel_audio))
        .fallback_service(ServeDir::new(static_dir))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding web server to {bind}"))?;
    info!("Web listener serving on {bind} - open this address in a browser");

    let handle = tokio::spawn(async move {
        let server = axum::serve(listener, router).with_graceful_shutdown(async move {
            cancel.cancelled().await;
        });
        if let Err(error) = server.await {
            warn!("web server exited with error: {error}");
        }
    });

    Ok(handle)
}
