use async_graphql::{Object, SimpleObject};

#[derive(SimpleObject)]
pub struct UpdateResponse {
    pub value: String,
}

pub struct Mutation;

#[Object]
impl Mutation {
    async fn update_something(&self, new_value: String) -> UpdateResponse {
        UpdateResponse {
            value: format!("Updated with: {}", new_value),
        }
    }
}
