use axum::{routing::get, serve, Router};
use http::Method;
use std::{env, time::Duration};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
};
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();

    #[cfg(debug_assertions)]
    dotenvy::from_filename(".env.development").ok();
    #[cfg(not(debug_assertions))]
    dotenvy::from_filename(".env.production").ok();

    let port = env::var("PORT")
        .expect("PORT must be set in environment")
        .parse::<u16>()
        .expect("PORT must be a valid number");

    let compression = CompressionLayer::new();
    let timeout = TimeoutLayer::new(Duration::from_secs(10));
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_origin(Any)
        .allow_headers(Any);
    let trace = TraceLayer::new_for_http();

    let app = Router::new()
        .route("/", get(root))
        .layer(compression)
        .layer(timeout)
        .layer(cors)
        .layer(trace);

    let listener = TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    info!("Server listening on port {}", port);

    serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello, World!"
}
