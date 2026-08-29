use async_graphql::SimpleObject;

#[derive(SimpleObject)]
pub struct StartWorkflowResponse {
    pub workflow_id: String,
}
