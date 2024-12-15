use axum::extract::Multipart;
use axum::Json;
use image::imageops::FilterType;
use serde::Serialize;
use std::{fs, path::Path};
use tch::{CModule, Device, Kind, Tensor};

const MODEL_PATH: &str = "models";
const WEIGHTS_FILE_NAME: &str = "resnet18.ot";
const LABELS_FILE_NAME: &str = "labels.txt";
const IMAGE_SIZE: u32 = 224;

#[derive(Serialize)]
pub struct ClassificationResponse {
    class_name: String,
    confidence: f64,
}

pub async fn root() -> &'static str {
    "ok"
}

async fn ensure_model_dir() -> Result<(), String> {
    if !Path::new(MODEL_PATH).exists() {
        tokio::fs::create_dir_all(MODEL_PATH)
            .await
            .map_err(|e| format!("Failed to create model directory: {}", e))?;
    }
    Ok(())
}

fn load_labels() -> Result<Vec<String>, String> {
    let content = fs::read_to_string(format!("{}/{}", MODEL_PATH, LABELS_FILE_NAME))
        .map_err(|e| format!("Failed to read labels file: {}", e))?;
    Ok(content.lines().map(String::from).collect())
}

fn process_image(image_data: &[u8]) -> Result<Tensor, String> {
    // Load image from bytes
    let img =
        image::load_from_memory(image_data).map_err(|e| format!("Failed to load image: {}", e))?;

    // Resize maintaining aspect ratio so that the smallest side is 256
    let (width, height) = (img.width(), img.height());
    let scale = 256.0 / width.min(height) as f32;
    let new_width = (width as f32 * scale).round() as u32;
    let new_height = (height as f32 * scale).round() as u32;
    let mut resized = img.resize(new_width, new_height, FilterType::Triangle);

    // Center crop to 224x224
    let left = (new_width.saturating_sub(IMAGE_SIZE)) / 2;
    let top = (new_height.saturating_sub(IMAGE_SIZE)) / 2;
    let cropped = resized.crop(left, top, IMAGE_SIZE, IMAGE_SIZE);

    // Convert to RGB if not already
    let rgb_img = cropped.to_rgb8();

    // Convert to float tensor and normalize
    let mut r_channel = Vec::with_capacity((IMAGE_SIZE * IMAGE_SIZE) as usize);
    let mut g_channel = Vec::with_capacity((IMAGE_SIZE * IMAGE_SIZE) as usize);
    let mut b_channel = Vec::with_capacity((IMAGE_SIZE * IMAGE_SIZE) as usize);

    // First separate into channels
    for pixel in rgb_img.pixels() {
        r_channel.push((pixel[0] as f32) / 255.0);
        g_channel.push((pixel[1] as f32) / 255.0);
        b_channel.push((pixel[2] as f32) / 255.0);
    }

    // Combine channels in correct order
    let mut tensor_data = Vec::with_capacity((IMAGE_SIZE * IMAGE_SIZE * 3) as usize);
    tensor_data.extend_from_slice(&r_channel);
    tensor_data.extend_from_slice(&g_channel);
    tensor_data.extend_from_slice(&b_channel);

    // Create tensor with shape [1, 3, 224, 224]
    let tensor = Tensor::from_slice(&tensor_data)
        .to_kind(Kind::Float)
        .view((1, 3, IMAGE_SIZE as i64, IMAGE_SIZE as i64))
        .to_device(Device::Cpu);

    // Normalize using ImageNet mean and std
    let mean = Tensor::from_slice(&[0.485_f32, 0.456_f32, 0.406_f32])
        .to_kind(Kind::Float)
        .view((1, 3, 1, 1))
        .to_device(Device::Cpu);
    let std = Tensor::from_slice(&[0.229_f32, 0.224_f32, 0.225_f32])
        .to_kind(Kind::Float)
        .view((1, 3, 1, 1))
        .to_device(Device::Cpu);

    Ok((tensor - mean) / std)
}

pub async fn classify_image_resnet(
    mut multipart: Multipart,
) -> Result<Json<ClassificationResponse>, String> {
    ensure_model_dir().await?;

    // Get image from multipart form
    let mut image_data = Vec::new();
    while let Some(field) = multipart.next_field().await.map_err(|e| e.to_string())? {
        if field.name().unwrap_or("") == "image" {
            let bytes = field.bytes().await.map_err(|e| e.to_string())?;
            image_data = bytes.to_vec();
            break;
        }
    }

    if image_data.is_empty() {
        return Err("No image found in request".to_string());
    }

    // Process image
    let image_tensor = process_image(&image_data)?;

    // Load the labels
    let labels = load_labels()?;

    // Load the TorchScript model and convert to float
    let model = CModule::load(format!("{}/{}", MODEL_PATH, WEIGHTS_FILE_NAME)).map_err(|e| format!("Failed to load model: {}", e))?;

    // Run inference
    let output = model
        .forward_ts(&[image_tensor])
        .map_err(|e| format!("Failed to run inference: {}", e))?;

    let output = output.to_kind(Kind::Float).softmax(-1, Kind::Float);

    // Get the top prediction
    let (confidence, class_index) = output.max_dim(1, true);

    // Convert tensors to scalar values
    let class_idx = class_index.int64_value(&[]) as usize;
    let confidence_val = confidence.double_value(&[]);

    let class_name = labels
        .get(class_idx)
        .map(String::from)
        .unwrap_or_else(|| format!("class_{}", class_idx));

    Ok(Json(ClassificationResponse {
        class_name,
        confidence: confidence_val,
    }))
}

#[cfg(test)]
#[path = "root_test.rs"]
mod tests;
