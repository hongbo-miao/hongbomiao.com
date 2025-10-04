use async_graphql::{Context, Object, SimpleObject, Upload};
use serde::Serialize;
use std::io::Read;
use utoipa::ToSchema;

use crate::shared::image::utils::load_labels::load_labels;
use crate::shared::image::utils::load_model::load_model;
use crate::shared::image::utils::process_image::process_image;

#[derive(SimpleObject, ToSchema)]
pub struct UpdateResponse {
    pub value: String,
}

#[derive(SimpleObject, Serialize, ToSchema)]
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
        let upload_value = image.value(ctx).map_err(|error| error.to_string())?;
        let mut image_data = Vec::new();
        upload_value
            .into_read()
            .read_to_end(&mut image_data)
            .map_err(|error| error.to_string())?;

        // Process image
        let image_data_processed = process_image(&image_data)?;

        // Run inference
        let session = load_model()?;

        // Create input tensor for ONNX Runtime
        let input_array = ndarray::CowArray::from(
            ndarray::Array4::from_shape_vec((1, 3, 224, 224), image_data_processed)
                .map_err(|error| format!("Failed to create input array: {}", error))?,
        )
        .into_dyn();

        let input_tensor = ort::Value::from_array(session.allocator(), &input_array)
            .map_err(|error| format!("Failed to create ONNX tensor: {}", error))?;

        let outputs = session
            .run(vec![input_tensor])
            .map_err(|error| format!("Failed to run inference: {}", error))?;

        // Get output tensor
        let output_tensor = outputs[0]
            .try_extract::<f32>()
            .map_err(|error| format!("Failed to extract output: {}", error))?;
        let output = output_tensor.view();

        // Apply softmax and find max
        let output_slice = output.as_slice().unwrap();
        let mut softmax_output = vec![0.0f32; output_slice.len()];

        // Compute softmax
        let max_val = output_slice
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = output_slice.iter().map(|&x| (x - max_val).exp()).sum();
        for (i, &val) in output_slice.iter().enumerate() {
            softmax_output[i] = (val - max_val).exp() / sum;
        }

        // Find the class with highest probability
        let (class_idx, &confidence_val) = softmax_output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let labels = load_labels()?;
        let class_name = labels
            .get(class_idx)
            .map(String::from)
            .unwrap_or_else(|| format!("class_{}", class_idx));

        Ok(ClassificationResponse {
            class_name,
            confidence: confidence_val as f64,
        })
    }
}
