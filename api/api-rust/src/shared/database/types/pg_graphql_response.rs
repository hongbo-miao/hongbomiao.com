use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Serialize, ToSchema)]
pub struct PgGraphqlResponse {
    pub data: Option<serde_json::Value>,
    pub errors: Option<Vec<serde_json::Value>>,
}
