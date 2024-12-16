use axum::extract::Multipart;
use axum::Json;
use opencv::core::{Mat, MatTraitConst, Size, CV_32F};
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::imgproc::{resize, INTER_LINEAR};
use serde::Serialize;
use std::fs;
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

fn load_labels() -> Result<Vec<String>, String> {
    let content = fs::read_to_string(format!("{}/{}", MODEL_PATH, LABELS_FILE_NAME))
        .map_err(|e| format!("Failed to read labels file: {}", e))?;
    Ok(content.lines().map(String::from).collect())
}

fn process_image(image_data: &[u8]) -> Result<Tensor, String> {
    // Load image from bytes using OpenCV
    let img_vec = Mat::from_slice(image_data)
        .map_err(|e| format!("Failed to create Mat from bytes: {}", e))?;
    let img = opencv::imgcodecs::imdecode(&img_vec, IMREAD_COLOR)
        .map_err(|e| format!("Failed to decode image: {}", e))?;

    // Resize maintaining aspect ratio so that the smallest side is 256
    let (width, height) = (img.cols(), img.rows());
    let scale = 256.0 / width.min(height) as f32;
    let new_width = (width as f32 * scale).round() as i32;
    let new_height = (height as f32 * scale).round() as i32;
    let mut resized = Mat::default();
    resize(
        &img,
        &mut resized,
        Size::new(new_width, new_height),
        0.0,
        0.0,
        INTER_LINEAR,
    )
    .map_err(|e| format!("Failed to resize image: {}", e))?;

    // Center crop to 224x224
    let left = (new_width.saturating_sub(IMAGE_SIZE as i32)) / 2;
    let top = (new_height.saturating_sub(IMAGE_SIZE as i32)) / 2;
    let roi = opencv::core::Rect::new(left, top, IMAGE_SIZE as i32, IMAGE_SIZE as i32);
    let mut cropped_mat = Mat::default();
    let roi_mat = resized
        .roi(roi)
        .map_err(|e| format!("Failed to get ROI: {}", e))?;
    roi_mat
        .copy_to(&mut cropped_mat)
        .map_err(|e| format!("Failed to crop image: {}", e))?;

    // Convert to float and normalize
    let mut float_mat = Mat::default();
    cropped_mat
        .convert_to(&mut float_mat, CV_32F, 1.0 / 255.0, 0.0)
        .map_err(|e| format!("Failed to convert to float: {}", e))?;

    // Split channels and create tensor
    let mut channels = opencv::core::Vector::<Mat>::new();
    opencv::core::split(&float_mat, &mut channels)
        .map_err(|e| format!("Failed to split channels: {}", e))?;

    let mut tensor_data = Vec::with_capacity((IMAGE_SIZE * IMAGE_SIZE * 3) as usize);
    for i in 0..3 {
        let channel = channels.get(i).unwrap();
        let mut channel_data = vec![0f32; (IMAGE_SIZE * IMAGE_SIZE) as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                channel.data() as *const f32,
                channel_data.as_mut_ptr(),
                channel_data.len(),
            );
        }
        tensor_data.extend_from_slice(&channel_data);
    }

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
    let model = CModule::load(format!("{}/{}", MODEL_PATH, WEIGHTS_FILE_NAME))
        .map_err(|e| format!("Failed to load model: {}", e))?;

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
