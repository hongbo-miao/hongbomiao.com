use anyhow::Result;
use opencv::core::{AlgorithmHint, MatTraitConst};
use opencv::imgcodecs::{IMREAD_COLOR, imread};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color};
use opencv::prelude::MatTraitConstManual;
use rerun as rr;
use std::path::Path;

pub fn log_camera_to_rerun<P: AsRef<Path>>(
    recording: &rr::RecordingStream,
    camera_image_path: P,
    entity_path: &str,
) -> Result<()> {
    // Load image using OpenCV
    let image = imread(
        camera_image_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path"))?,
        IMREAD_COLOR,
    )?;

    if image.empty() {
        anyhow::bail!(
            "Failed to load camera image from {}",
            camera_image_path.as_ref().display()
        );
    }

    // Convert BGR to RGB
    let mut rgb_image = opencv::core::Mat::default();
    cvt_color(
        &image,
        &mut rgb_image,
        COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Get image dimensions
    let height = rgb_image.rows() as u32;
    let width = rgb_image.cols() as u32;

    // Get image data as slice
    let image_data = rgb_image.data_bytes()?;

    // Log to Rerun
    recording.log(
        entity_path,
        &rr::Image::from_rgb24(image_data, [width, height]),
    )?;

    Ok(())
}
