use crate::config::AppConfig;
use crate::shared::camera::services::detect_objects_in_camera::{
    YoloModel, detect_objects_in_camera,
};
use crate::shared::fusion::constants::colors::{COLOR_BLACK_SCALAR, COLOR_CORAL_RED_SCALAR};
use anyhow::{Context, Result};
use opencv::core::{Point, Rect, Scalar};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::imread;
use opencv::imgproc::{HersheyFonts, LINE_8, get_text_size, put_text, rectangle};
use opencv::prelude::*;
use std::path::Path;
use tracing::info;

pub fn visualize_camera_only<P: AsRef<Path>>(
    camera_image_path: P,
    yolo_model: &mut YoloModel,
    _config: AppConfig,
) -> Result<bool> {
    let image = imread(
        camera_image_path
            .as_ref()
            .to_str()
            .expect("Failed to convert camera image path to string"),
        opencv::imgcodecs::IMREAD_COLOR,
    )
    .context("Failed to load camera image")?;

    if image.empty() {
        return Ok(true);
    }

    let camera_detections = detect_objects_in_camera(&image, yolo_model, 0.6)?;

    let mut visualization = image.clone();

    for camera_detection in &camera_detections {
        let bounding_box = camera_detection.bounding_box;
        let x1 = bounding_box[0] as i32;
        let y1 = bounding_box[1] as i32;
        let x2 = bounding_box[2] as i32;
        let y2 = bounding_box[3] as i32;

        rectangle(
            &mut visualization,
            Rect::new(x1, y1, x2 - x1, y2 - y1),
            COLOR_CORAL_RED_SCALAR,
            2,
            LINE_8,
            0,
        )?;

        let label = format!(
            "{} {:.0}%",
            camera_detection.class_name,
            camera_detection.confidence * 100.0
        );

        let text_size = get_text_size(
            &label,
            HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
            0.4,
            1,
            &mut 0,
        )?;

        rectangle(
            &mut visualization,
            Rect::new(
                x1,
                y1 - text_size.height - 4,
                text_size.width,
                text_size.height + 4,
            ),
            COLOR_CORAL_RED_SCALAR,
            -1,
            LINE_8,
            0,
        )?;

        put_text(
            &mut visualization,
            &label,
            Point::new(x1, y1 - 2),
            HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
            0.4,
            COLOR_BLACK_SCALAR,
            1,
            LINE_8,
            false,
        )?;
    }

    let info_text = format!("Camera-only: {} detections", camera_detections.len());
    put_text(
        &mut visualization,
        &info_text,
        Point::new(10, 30),
        HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
        0.4,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        LINE_8,
        false,
    )?;

    let controls_text = "Controls: [Space] Pause/Resume | [q] Quit";
    put_text(
        &mut visualization,
        controls_text,
        Point::new(10, 60),
        HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
        0.3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        LINE_8,
        false,
    )?;

    imshow("Camera Only", &visualization)?;
    let key = wait_key(50)?;
    if key == 'q' as i32 {
        info!("Visualization stopped by user request");
        return Ok(false);
    } else if key == ' ' as i32 {
        info!("Paused - Press [Space] to resume or [q] to quit");
        loop {
            let pause_key = wait_key(0)?;
            if pause_key == ' ' as i32 {
                info!("Resumed");
                break;
            } else if pause_key == 'q' as i32 {
                info!("Visualization stopped by user request");
                return Ok(false);
            }
        }
    }

    info!("Visualization complete");

    Ok(true)
}
