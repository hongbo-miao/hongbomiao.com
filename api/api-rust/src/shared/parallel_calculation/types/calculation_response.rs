use async_graphql::SimpleObject;
use serde::Serialize;
use utoipa::ToSchema;

#[derive(SimpleObject, Serialize, ToSchema)]
pub struct CalculationResponse {
    pub result: f64,
    pub duration_milliseconds: u64,
    pub items_processed: i32,
    pub thread_count: i32,
}
