use axum::Json;

use crate::agent::constants::agent_model_id::AGENT_MODEL_ID;
use crate::agent::types::model_info::ModelInfo;
use crate::agent::types::model_list_response::ModelListResponse;

/// GET /v1/models
pub async fn handle_list_models_request() -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: AGENT_MODEL_ID.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "fluss-demo".to_string(),
        }],
    })
}
