use async_graphql::Schema;
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{extract::State, response::IntoResponse};
use sqlx::PgPool;

use super::mutation::Mutation;
use super::query::Query;
use super::subscription::Subscription;
use crate::shared::application::types::application_state::ApplicationState;

pub type ApiSchema = Schema<Query, Mutation, Subscription>;

pub async fn graphql_handler(
    State(application_state): State<ApplicationState>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    application_state
        .schema
        .execute(req.into_inner())
        .await
        .into()
}

pub async fn graphiql() -> impl IntoResponse {
    axum::response::Html(
        async_graphql::http::GraphiQLSource::build()
            .endpoint("/graphql")
            .subscription_endpoint("/ws")
            .finish(),
    )
}

pub fn create_schema(pool: PgPool) -> ApiSchema {
    Schema::build(Query, Mutation, Subscription)
        .data(pool)
        .limit_depth(100)
        .limit_complexity(1000)
        .finish()
}
