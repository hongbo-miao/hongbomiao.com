use anyhow::Result;
use opencv::core::{AlgorithmHint, Mat, MatTraitConst, MatTraitConstManual};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color};
use rerun as rr;

pub fn log_rerun_image(
    recording: &rr::RecordingStream,
    image: &Mat,
    entity_path: &str,
) -> Result<()> {
    let mut rgb_image = Mat::default();
    cvt_color(
        image,
        &mut rgb_image,
        COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let height = rgb_image.rows() as u32;
    let width = rgb_image.cols() as u32;
    let image_data = rgb_image.data_bytes()?;

    recording.log(
        entity_path,
        &rr::Image::from_rgb24(image_data, [width, height]),
    )?;

    Ok(())
}
