pub mod resolvers;
pub mod schema;
use self::schema::Query;
use async_graphql::{EmptyMutation, EmptySubscription, Schema};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::extract::State;
use axum::response::IntoResponse;

pub type ApiSchema = Schema<Query, EmptyMutation, EmptySubscription>;

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
            .finish(),
    )
}

pub fn create_schema() -> ApiSchema {
    Schema::new(Query, EmptyMutation, EmptySubscription)
}
