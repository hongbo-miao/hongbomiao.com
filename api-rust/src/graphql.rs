mod mutation;
mod query;
mod subscription;
use self::mutation::Mutation;
use self::query::Query;
use self::subscription::Subscription;
use async_graphql::Schema;
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{extract::State, response::IntoResponse};

type ApiSchema = Schema<Query, Mutation, Subscription>;

pub async fn graphql_handler(
    State(schema): State<ApiSchema>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

pub async fn graphiql() -> impl IntoResponse {
    axum::response::Html(
        async_graphql::http::GraphiQLSource::build()
            .endpoint("/graphql")
            .subscription_endpoint("/ws")
            .finish(),
    )
}

pub fn create_schema() -> ApiSchema {
    Schema::build(Query, Mutation, Subscription)
        .limit_depth(100)
        .limit_complexity(1000)
        .finish()
}
