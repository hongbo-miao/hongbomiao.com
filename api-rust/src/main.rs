mod graphql;
mod handlers;
use axum::routing::{get, post};
use axum::Router;
use graphql::{create_schema, graphiql, graphql_handler};
use http::Method;
use std::net::SocketAddr;
use std::sync::Arc;
use std::{env, time::Duration};
use tower_governor::governor::GovernorConfigBuilder;
use tower_governor::key_extractor::SmartIpKeyExtractor;
use tower_governor::GovernorLayer;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
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
    let governor_conf = Arc::new(
        GovernorConfigBuilder::default()
            .per_second(20)
            .burst_size(50)
            .key_extractor(SmartIpKeyExtractor)
            .finish()
            .unwrap(),
    );

    let schema = create_schema();

    let app = Router::new()
        .route("/", get(handlers::root::root))
        .route(
            "/classify-image",
            post(handlers::root::classify_image_resnet),
        )
        .route("/graphiql", get(graphiql))
        .route("/graphql", post(graphql_handler))
        .route_service(
            "/ws",
            async_graphql_axum::GraphQLSubscription::new(schema.clone()),
        )
        .with_state(schema)
        .layer(compression)
        .layer(timeout)
        .layer(cors)
        .layer(trace)
        .layer(GovernorLayer {
            config: governor_conf,
        });

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}
