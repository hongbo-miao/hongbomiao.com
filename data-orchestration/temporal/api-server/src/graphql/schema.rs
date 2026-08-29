use async_graphql::{EmptySubscription, Schema};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use temporalio_client::Client;

use crate::graphql::mutation::Mutation;
use crate::graphql::query::Query;
use crate::types::application_state::ApplicationState;

pub type ApiSchema = Schema<Query, Mutation, EmptySubscription>;

pub fn create_schema(temporal_client: Client) -> ApiSchema {
    Schema::build(Query, Mutation, EmptySubscription)
        .data(temporal_client)
        .finish()
}

pub async fn graphql_handler(
    State(application_state): State<ApplicationState>,
    request: GraphQLRequest,
) -> GraphQLResponse {
    application_state
        .schema
        .execute(request.into_inner())
        .await
        .into()
}

pub async fn graphiql() -> impl IntoResponse {
    Html(
        async_graphql::http::GraphiQLSource::build()
            .endpoint("/graphql")
            .finish(),
    )
}
