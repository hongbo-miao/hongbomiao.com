use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use rig::agent::Agent;
use rig::providers::openrouter::completion::CompletionModel;

use crate::agent::handlers::handle_chat_completions_request::handle_chat_completions_request;
use crate::agent::handlers::handle_list_models_request::handle_list_models_request;

pub async fn serve_agent_endpoints(agent: Arc<Agent<CompletionModel>>) -> anyhow::Result<()> {
    let router = Router::new()
        .route(
            "/v1/chat/completions",
            post(handle_chat_completions_request),
        )
        .route("/v1/models", get(handle_list_models_request))
        .with_state(agent);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8083").await?;
    axum::serve(listener, router).await?;
    Ok(())
}
