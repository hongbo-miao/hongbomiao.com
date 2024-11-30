use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();

    let app = axum::Router::new().route("/", axum::routing::get(root));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:36147")
        .await
        .unwrap();
    info!("Server listening on port 36147");

    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello, World!"
}
