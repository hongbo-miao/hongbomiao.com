#![forbid(unsafe_code)]

mod config;
mod graphql;
mod handlers;
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

use crate::config::AppConfig;
use crate::graphql::schema;

#[tokio::main]
async fn main() {
    let config = AppConfig::get();

    tracing_subscriber::fmt()
        .with_max_level(config.log_level)
        .init();
    let schema = schema::create_schema();
    let compression = CompressionLayer::new();
    let timeout = TimeoutLayer::new(Duration::from_secs(10));
    let allowed_origins: Vec<HeaderValue> = config
        .cors_allowed_origins
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
            .per_second(20)
            .burst_size(50)
            .key_extractor(SmartIpKeyExtractor)
            .finish()
            .unwrap(),
    ));

    let app = Router::new()
        .route("/", get(handlers::root::root))
        .route("/graphiql", get(schema::graphiql))
        .route("/graphql", post(schema::graphql_handler))
        .route(
            "/ws/police-audio-stream",
            get(handlers::police_audio_stream::get_police_audio_stream),
        )
        .route_service(
            "/ws",
            async_graphql_axum::GraphQLSubscription::new(schema.clone()),
        )
        .with_state(schema)
        .layer(compression)
        .layer(timeout)
        .layer(cors)
        .layer(trace)
        .layer(governor);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}
