use async_graphql::{Object, SimpleObject};

#[derive(SimpleObject)]
pub struct HelloResponse {
    pub message: String,
}

pub struct Query;

#[Object]
impl Query {
    async fn hello(&self) -> HelloResponse {
        HelloResponse {
            message: "Hello, world!".to_string(),
        }
    }
}
