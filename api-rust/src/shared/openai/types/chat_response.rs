use async_graphql::SimpleObject;

#[derive(SimpleObject)]
pub struct ChatResponse {
    pub content: String,
}
