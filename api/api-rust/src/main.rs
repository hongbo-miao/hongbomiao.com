#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod graphql;
mod handlers;
mod openapi;
mod shared;
mod webtransport;

use anyhow::Result;
use axum::Router;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::routing::{get, post};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tower_governor::GovernorLayer;
use tower_governor::governor::GovernorConfigBuilder;
use tower_governor::key_extractor::SmartIpKeyExtractor;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

use crate::openapi::ApiDoc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::config::AppConfig;
use crate::graphql::schema;
use crate::shared::application::types::application_state::ApplicationState;
use crate::shared::database::utils::initialize_pool::initialize_pool;
use crate::webtransport::services::webtransport_server::WebTransportServer;

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    tracing_subscriber::fmt()
        .with_max_level(config.server_log_level)
        .init();
    ffmpeg_sidecar::download::auto_download()?;

    let pool = initialize_pool(&config.postgres_url, config.postgres_max_connection_count).await?;

    // Start WebTransport server in parallel
    let webtransport_port = config.server_port + 1;
    let webtransport_server = WebTransportServer::create(webtransport_port).await?;
    let webtransport_handle = tokio::spawn(async move {
        if let Err(error) = webtransport_server.serve().await {
            error!("WebTransport server error: {error}");
        }
    });

    let schema = schema::create_schema(pool.clone());
    let application_state = ApplicationState {
        schema: schema.clone(),
    };
    let compression = CompressionLayer::new();
    let timeout =
        TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, Duration::from_secs(30));
    let allowed_origins: Vec<HeaderValue> = config
        .server_cors_allowed_origins
        .iter()
        .filter_map(|origin| HeaderValue::from_str(origin).ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::HEAD,
            Method::OPTIONS,
            Method::PATCH,
            Method::POST,
            Method::PUT,
        ])
        .allow_origin(allowed_origins)
        .allow_headers(Any);
    let trace = TraceLayer::new_for_http();
    let governor = GovernorLayer::new(Arc::new(
        GovernorConfigBuilder::default()
            .per_second(config.server_rate_limit_per_second.into())
            .burst_size(config.server_rate_limit_per_second_burst.into())
            .key_extractor(SmartIpKeyExtractor)
            .finish()
            .unwrap(),
    ));

    // Routes that need timeout protection
    let timeout_routes = Router::new()
        .route("/", get(handlers::get_root::get_root))
        .route("/graphiql", get(schema::graphiql))
        .route("/graphql", post(schema::graphql_handler))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(timeout);

    // Long-lived connection routes (SSE, WebSocket) without timeout
    let streaming_routes = Router::new()
        .route("/sse/events", get(handlers::get_sse_events::get_sse_events))
        .route(
            "/ws/emergency-audio-stream",
            get(handlers::get_emergency_audio_stream::get_emergency_audio_stream),
        )
        .route(
            "/ws/police-audio-stream",
            get(handlers::get_police_audio_stream::get_police_audio_stream),
        )
        .route_service(
            "/ws",
            async_graphql_axum::GraphQLSubscription::new(schema.clone()),
        );

    let app = Router::new()
        .merge(timeout_routes)
        .merge(streaming_routes)
        .with_state(application_state)
        .layer(compression)
        .layer(cors)
        .layer(trace)
        .layer(governor);

    let address = format!("[::]:{}", config.server_port).parse::<SocketAddr>()?;
    info!("HTTP server listening on {}", address);
    info!(
        "WebTransport server listening on port {}",
        webtransport_port
    );
    let listener = tokio::net::TcpListener::bind(address).await?;

    // Run both servers concurrently
    tokio::select! {
        result = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        ) => {
            if let Err(error) = result {
                error!("Axum server error: {}", error);
            }
        }
        _ = webtransport_handle => {
            error!("WebTransport server stopped unexpectedly");
        }
    }

    Ok(())
}
