use async_graphql::{Context, Object, SimpleObject, Upload};
use serde::Serialize;
use std::io::Read;

use crate::graphql::utils::resnet_18_util;

#[derive(SimpleObject)]
pub struct UpdateResponse {
    pub value: String,
}

#[derive(SimpleObject, Serialize)]
pub struct ClassificationResponse {
    pub class_name: String,
    pub confidence: f64,
}

pub struct Mutation;

#[Object]
impl Mutation {
    async fn update_something(&self, new_value: String) -> UpdateResponse {
        UpdateResponse {
            value: format!("Updated with: {}", new_value),
        }
    }

    async fn classify_image(
        &self,
        ctx: &Context<'_>,
        image: Upload,
    ) -> Result<ClassificationResponse, String> {
        // Get image data from upload
        let upload_value = image.value(ctx).map_err(|e| e.to_string())?;
        let mut image_data = Vec::new();
        upload_value
            .into_read()
            .read_to_end(&mut image_data)
            .map_err(|e| e.to_string())?;

        // Process image
        let image_tensor = resnet_18_util::process_image(&image_data)?;

        // Run inference
        let model = resnet_18_util::load_model()?;
        let output = model
            .forward_ts(&[image_tensor])
            .map_err(|e| format!("Failed to run inference: {}", e))?;

        let output = output
            .to_kind(tch::Kind::Float)
            .softmax(-1, tch::Kind::Float);

        // Get the top prediction
        let (confidence, class_index) = output.max_dim(1, true);

        // Convert tensors to scalar values
        let class_idx = class_index.int64_value(&[]) as usize;
        let confidence_val = confidence.double_value(&[]);

        let labels = resnet_18_util::load_labels()?;
        let class_name = labels
            .get(class_idx)
            .map(String::from)
            .unwrap_or_else(|| format!("class_{}", class_idx));

        Ok(ClassificationResponse {
            class_name,
            confidence: confidence_val,
        })
    }
}
