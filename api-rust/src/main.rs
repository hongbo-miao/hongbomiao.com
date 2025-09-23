#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod graphql;
mod handlers;
mod openapi;
mod shared;

use axum::Router;
use axum::http::{HeaderValue, Method};
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
use tracing::info;

use crate::openapi::ApiDoc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::config::AppConfig;
use crate::graphql::schema;

#[tokio::main]
async fn main() {
    let config = AppConfig::get();
    tracing_subscriber::fmt()
        .with_max_level(config.server_log_level)
        .init();
    ffmpeg_sidecar::download::auto_download().expect("Failed to download FFmpeg");

    let schema = schema::create_schema();
    let compression = CompressionLayer::new();
    let timeout = TimeoutLayer::new(Duration::from_secs(30));
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
            .per_second(config.server_rate_limit_per_second)
            .burst_size(config.server_rate_limit_per_second_burst)
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
        .with_state(schema)
        .layer(compression)
        .layer(cors)
        .layer(trace)
        .layer(governor);

    let address = format!("[::]:{}", config.server_port)
        .parse::<SocketAddr>()
        .expect("Failed to parse socket address");
    info!("Listening on {}", address);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("Failed to bind TCP listener");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}
