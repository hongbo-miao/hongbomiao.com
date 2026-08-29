use axum::Router;
use axum::routing::{get, post};

use crate::graphql::schema::{graphiql, graphql_handler};
use crate::handlers::get_root::get_root;
use crate::types::application_state::ApplicationState;

pub async fn serve_api_endpoints(application_state: ApplicationState) -> anyhow::Result<()> {
    let router = Router::new()
        .route("/", get(get_root))
        .route("/graphql", post(graphql_handler))
        .route("/graphiql", get(graphiql))
        .with_state(application_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8085").await?;
    axum::serve(listener, router).await?;
    Ok(())
}
