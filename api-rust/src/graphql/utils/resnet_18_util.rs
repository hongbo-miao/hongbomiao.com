use opencv::core::{Mat, MatTraitConst, Size, CV_32F};
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::imgproc::{resize, INTER_LINEAR};
use opencv::prelude::MatTraitConstManual;
use std::fs;
use tch::{CModule, Device, Kind, Tensor};

const MODEL_PATH: &str = "models";
const WEIGHTS_FILE_NAME: &str = "resnet18.ot";
const LABELS_FILE_NAME: &str = "labels.txt";
const IMAGE_SIZE: u32 = 224;

pub fn load_labels() -> Result<Vec<String>, String> {
    let content = fs::read_to_string(format!("{}/{}", MODEL_PATH, LABELS_FILE_NAME))
        .map_err(|e| format!("Failed to read labels file: {}", e))?;
    Ok(content.lines().map(String::from).collect())
}

pub fn load_model() -> Result<CModule, String> {
    let model_path = format!("{}/{}", MODEL_PATH, WEIGHTS_FILE_NAME);
    let model = CModule::load(&model_path)
        .map_err(|e| format!("Failed to load model from {}: {}", model_path, e))?;
    Ok(model)
}

pub fn process_image(image_data: &[u8]) -> Result<Tensor, String> {
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
        let channel_data = channel
            .data_typed::<f32>()
            .map_err(|e| format!("Failed to get typed data: {}", e))?
            .to_vec();
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
