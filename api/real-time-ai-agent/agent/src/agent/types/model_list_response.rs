use serde::Serialize;

use crate::agent::types::model_info::ModelInfo;

#[derive(Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}
