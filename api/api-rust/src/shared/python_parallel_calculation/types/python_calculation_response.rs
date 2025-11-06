use async_graphql::SimpleObject;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(SimpleObject, Serialize, Deserialize, ToSchema)]
pub struct PythonCalculationResponse {
    pub result: f64,
    pub duration_milliseconds: u64,
    pub items_processed: i32,
    pub thread_count: i32,
}
