use opencv::core::{CV_32F, Mat, MatTraitConst, Size};
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::imgproc::{INTER_LINEAR, resize};
use opencv::prelude::MatTraitConstManual;

const IMAGE_SIZE: u32 = 224;

pub fn process_image(image_data: &[u8]) -> Result<Vec<f32>, String> {
    // Load image from bytes using OpenCV
    let img_vec = Mat::from_slice(image_data)
        .map_err(|error| format!("Failed to create Mat from bytes: {}", error))?;
    let img = opencv::imgcodecs::imdecode(&img_vec, IMREAD_COLOR)
        .map_err(|error| format!("Failed to decode image: {}", error))?;

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
    .map_err(|error| format!("Failed to resize image: {}", error))?;

    // Center crop to 224x224
    let left = (new_width.saturating_sub(IMAGE_SIZE as i32)) / 2;
    let top = (new_height.saturating_sub(IMAGE_SIZE as i32)) / 2;
    let roi = opencv::core::Rect::new(left, top, IMAGE_SIZE as i32, IMAGE_SIZE as i32);
    let mut cropped_mat = Mat::default();
    let roi_mat = resized
        .roi(roi)
        .map_err(|error| format!("Failed to get ROI: {}", error))?;
    roi_mat
        .copy_to(&mut cropped_mat)
        .map_err(|error| format!("Failed to crop image: {}", error))?;

    // Convert to float and normalize
    let mut float_mat = Mat::default();
    cropped_mat
        .convert_to(&mut float_mat, CV_32F, 1.0 / 255.0, 0.0)
        .map_err(|error| format!("Failed to convert to float: {}", error))?;

    // Split channels and create tensor
    let mut channels = opencv::core::Vector::<Mat>::new();
    opencv::core::split(&float_mat, &mut channels)
        .map_err(|error| format!("Failed to split channels: {}", error))?;

    let mut tensor_data = Vec::with_capacity((IMAGE_SIZE * IMAGE_SIZE * 3) as usize);
    for i in 0..3 {
        let channel = channels.get(i).unwrap();
        let channel_data = channel
            .data_typed::<f32>()
            .map_err(|error| format!("Failed to get typed data: {}", error))?
            .to_vec();
        tensor_data.extend_from_slice(&channel_data);
    }

    // Normalize using ImageNet mean and std
    let mean = [0.485_f32, 0.456_f32, 0.406_f32];
    let std = [0.229_f32, 0.224_f32, 0.225_f32];

    // Apply normalization channel-wise
    let pixels_per_channel = (IMAGE_SIZE * IMAGE_SIZE) as usize;
    for c in 0..3 {
        let channel_start = c * pixels_per_channel;
        for pixel in tensor_data
            .iter_mut()
            .skip(channel_start)
            .take(pixels_per_channel)
        {
            *pixel = (*pixel - mean[c]) / std[c];
        }
    }

    Ok(tensor_data)
}
