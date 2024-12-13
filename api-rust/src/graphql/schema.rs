use super::resolvers::hello;
use async_graphql::Object;

pub struct Query;

#[Object]
impl Query {
    async fn hello(&self) -> &str {
        hello::hello_world()
    }
}
