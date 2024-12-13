mod graphql;
use crate::graphql::{create_schema, graphiql, graphql_handler};
use axum::routing::{get, post};
use axum::{serve, Router};
use http::Method;
use std::{env, time::Duration};
use tokio::net::TcpListener;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

async fn root() -> &'static str {
    "ok"
}

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

    let schema = create_schema();

    let app = Router::new()
        .route("/", get(root))
        .route("/graphiql", get(graphiql))
        .route("/graphql", post(graphql_handler))
        .with_state(schema)
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
