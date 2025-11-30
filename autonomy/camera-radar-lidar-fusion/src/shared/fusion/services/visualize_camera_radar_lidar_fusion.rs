use crate::config::AppConfig;
use crate::shared::camera::services::detect_objects_in_camera::{
    YoloModel, detect_objects_in_camera,
};
use crate::shared::fusion::constants::colors::{
    COLOR_BLACK_SCALAR, COLOR_BLUE_SCALAR, COLOR_GREEN_SCALAR,
};
use crate::shared::fusion::services::fuse_camera_radar_lidar::fuse_camera_radar_lidar;
use crate::shared::fusion::utils::calculate_distance_color::calculate_distance_color;
use crate::shared::lidar::services::create_lidar_detection::create_lidar_detection;
use crate::shared::lidar::services::load_lidar_data::load_lidar_data;
use crate::shared::lidar::utils::project_lidar_to_camera::project_lidar_to_camera;
use crate::shared::radar::services::create_radar_detection::create_radar_detection;
use crate::shared::radar::services::load_radar_data::load_radar_data;
use crate::shared::radar::utils::project_radar_to_camera::project_radar_to_camera;
use crate::shared::rerun::constants::entity_paths::FUSION_PROJECTION_CAM_FRONT_ENTITY_PATH;
use crate::shared::rerun::services::log_rerun_image::log_rerun_image;
use anyhow::{Context, Result};
use nalgebra::{Matrix3, Matrix4, Vector3};
use opencv::core::{Point, Rect, Scalar};
use opencv::imgcodecs::imread;
use opencv::imgproc::{HersheyFonts, LINE_8, get_text_size, put_text, rectangle};
use opencv::prelude::MatTraitConst;
use rerun as rr;
use std::path::Path;
use tracing::warn;

#[allow(clippy::too_many_arguments)]
pub fn visualize_camera_radar_lidar_fusion<P: AsRef<Path>>(
    recording: &rr::RecordingStream,
    camera_image_path: P,
    radar_data_path: P,
    lidar_data_path: P,
    camera_intrinsic: Matrix3<f64>,
    radar_to_camera: Matrix4<f64>,
    lidar_to_camera: Matrix4<f64>,
    yolo_model: &mut YoloModel,
    config: AppConfig,
) -> Result<()> {
    let image = imread(
        camera_image_path
            .as_ref()
            .to_str()
            .expect("Failed to convert camera image path to string"),
        opencv::imgcodecs::IMREAD_COLOR,
    )
    .context("Failed to load camera image")?;

    if image.empty() {
        warn!("Failed to load image");
        return Ok(());
    }

    let radar_data = load_radar_data(radar_data_path)?;

    let radar_points_3d = radar_data.fixed_rows::<3>(0);
    let radar_velocities = radar_data.row(3);
    let radar_cross_sections = radar_data.row(4);

    let radar_points_vec: Vec<Vector3<f64>> = (0..radar_points_3d.ncols())
        .map(|column_index| {
            Vector3::new(
                radar_points_3d[(0, column_index)] as f64,
                radar_points_3d[(1, column_index)] as f64,
                radar_points_3d[(2, column_index)] as f64,
            )
        })
        .collect();

    let image_points =
        project_radar_to_camera(&radar_points_vec, &camera_intrinsic, &radar_to_camera);

    if image_points.is_empty() {
        warn!("No radar points projected into camera frame");
    }

    let lidar_data = load_lidar_data(lidar_data_path)?;

    let lidar_points_3d = lidar_data.fixed_rows::<3>(0);
    let lidar_intensities = lidar_data.row(3);

    let lidar_points_vec: Vec<Vector3<f64>> = (0..lidar_points_3d.ncols())
        .map(|column_index| {
            Vector3::new(
                lidar_points_3d[(0, column_index)] as f64,
                lidar_points_3d[(1, column_index)] as f64,
                lidar_points_3d[(2, column_index)] as f64,
            )
        })
        .collect();

    let lidar_image_points =
        project_lidar_to_camera(&lidar_points_vec, &camera_intrinsic, &lidar_to_camera);

    if lidar_image_points.is_empty() {
        warn!("No lidar points projected into camera frame");
    }

    let camera_detections = detect_objects_in_camera(&image, yolo_model, 0.6)?;

    let mut radar_detections = Vec::new();
    for index in 0..radar_points_3d.ncols() {
        if index >= image_points.len() {
            warn!(
                "Projection list shorter than radar points; skipping index {}",
                index
            );
            break;
        }
        let projected = match image_points[index] {
            Some(value) => value,
            None => continue,
        };
        let point_3d = Vector3::new(
            radar_points_3d[(0, index)] as f64,
            radar_points_3d[(1, index)] as f64,
            radar_points_3d[(2, index)] as f64,
        );
        let velocity = radar_velocities[index] as f64;
        let radar_cross_section = radar_cross_sections[index] as f64;
        let image_x = projected[0];
        let image_y = projected[1];

        radar_detections.push(create_radar_detection(
            point_3d,
            velocity,
            radar_cross_section,
            image_x,
            image_y,
        ));
    }

    let mut lidar_detections = Vec::new();
    for index in 0..lidar_points_3d.ncols() {
        if index >= lidar_image_points.len() {
            warn!(
                "Projection list shorter than lidar points; skipping index {}",
                index
            );
            break;
        }
        let projected = match lidar_image_points[index] {
            Some(value) => value,
            None => continue,
        };
        let point_3d = Vector3::new(
            lidar_points_3d[(0, index)] as f64,
            lidar_points_3d[(1, index)] as f64,
            lidar_points_3d[(2, index)] as f64,
        );
        let intensity = lidar_intensities[index] as f64;
        let image_x = projected[0];
        let image_y = projected[1];

        lidar_detections.push(create_lidar_detection(
            point_3d, intensity, image_x, image_y,
        ));
    }

    let fusion_result = fuse_camera_radar_lidar(
        camera_detections,
        radar_detections,
        lidar_detections,
        config.association_distance_threshold_pixels,
        &config,
    );

    let mut visualization = image.clone();

    // Draw lidar-only detections first (bottom layer)
    for (index, lidar_detection) in fusion_result.unmatched_lidar_detections.iter().enumerate() {
        let pixel_x = lidar_detection.image_coordinate_x as i32;
        let pixel_y = lidar_detection.image_coordinate_y as i32;

        if pixel_x >= 0
            && pixel_x < visualization.cols()
            && pixel_y >= 0
            && pixel_y < visualization.rows()
        {
            let color = calculate_distance_color(
                lidar_detection.distance,
                config.association_distance_threshold_pixels,
            );
            let radius = ((lidar_detection.intensity * 0.5) as i32).clamp(1, 2);
            opencv::imgproc::circle(
                &mut visualization,
                Point::new(pixel_x, pixel_y),
                radius,
                color,
                -1,
                LINE_8,
                0,
            )?;

            if index % 5 == 0 {
                put_text(
                    &mut visualization,
                    &format!("{:.1}m", lidar_detection.distance),
                    Point::new(pixel_x + 8, pixel_y + 8),
                    HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
                    0.3,
                    color,
                    1,
                    LINE_8,
                    false,
                )?;
            }
        }
    }

    // Draw radar-only detections (second layer)
    for (index, radar_detection) in fusion_result.unmatched_radar_detections.iter().enumerate() {
        let pixel_x = radar_detection.image_coordinate_x as i32;
        let pixel_y = radar_detection.image_coordinate_y as i32;

        if pixel_x >= 0
            && pixel_x < visualization.cols()
            && pixel_y >= 0
            && pixel_y < visualization.rows()
        {
            let color = calculate_distance_color(
                radar_detection.distance,
                config.association_distance_threshold_pixels,
            );
            let radius = ((radar_detection.radar_cross_section / 10.0) as i32).clamp(3, 10);
            opencv::imgproc::circle(
                &mut visualization,
                Point::new(pixel_x, pixel_y),
                radius,
                color,
                -1,
                LINE_8,
                0,
            )?;

            if index % 5 == 0 {
                put_text(
                    &mut visualization,
                    &format!("{:.1}m", radar_detection.distance),
                    Point::new(pixel_x + 8, pixel_y - 8),
                    HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
                    0.3,
                    color,
                    1,
                    LINE_8,
                    false,
                )?;
            }
        }
    }

    // Draw camera-only detections (third layer)
    for camera_detection in &fusion_result.unmatched_camera_detections {
        let bounding_box = camera_detection.bounding_box;
        let x1 = bounding_box[0] as i32;
        let y1 = bounding_box[1] as i32;
        let x2 = bounding_box[2] as i32;
        let y2 = bounding_box[3] as i32;

        rectangle(
            &mut visualization,
            Rect::new(x1, y1, x2 - x1, y2 - y1),
            COLOR_BLUE_SCALAR,
            2,
            LINE_8,
            0,
        )?;

        let label = format!("{} (cam only)", camera_detection.class_name);

        put_text(
            &mut visualization,
            &label,
            Point::new(x1, y1 - 5),
            HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
            0.3,
            COLOR_BLUE_SCALAR,
            1,
            LINE_8,
            false,
        )?;
    }

    // Draw fused tracks last (top layer)
    for track in &fusion_result.fused_tracks {
        let bounding_box = track.bounding_box();
        let x1 = bounding_box[0] as i32;
        let y1 = bounding_box[1] as i32;
        let x2 = bounding_box[2] as i32;
        let y2 = bounding_box[3] as i32;

        rectangle(
            &mut visualization,
            Rect::new(x1, y1, x2 - x1, y2 - y1),
            COLOR_GREEN_SCALAR,
            2,
            LINE_8,
            0,
        )?;

        let label = if track.is_moving(&config) {
            if let Some(lidar_distance) = track.lidar_distance() {
                if let Some(radar_distance) = track.radar_distance() {
                    format!(
                        "{} {:.1}m (Radar:{:.1}m, Lidar:{:.1}m), {:.1}m/s",
                        track.class_name(),
                        track.distance(),
                        radar_distance,
                        lidar_distance,
                        track.velocity()
                    )
                } else {
                    format!(
                        "{} {:.1}m (Lidar:{:.1}m), {:.1}m/s",
                        track.class_name(),
                        track.distance(),
                        lidar_distance,
                        track.velocity()
                    )
                }
            } else {
                format!(
                    "{} {:.1}m {:.1}m/s",
                    track.class_name(),
                    track.distance(),
                    track.velocity()
                )
            }
        } else if let Some(lidar_distance) = track.lidar_distance() {
            if let Some(radar_distance) = track.radar_distance() {
                format!(
                    "{} {:.1}m (Radar:{:.1}m, Lidar:{:.1}m)",
                    track.class_name(),
                    track.distance(),
                    radar_distance,
                    lidar_distance
                )
            } else {
                format!(
                    "{} {:.1}m (Lidar:{:.1}m)",
                    track.class_name(),
                    track.distance(),
                    lidar_distance
                )
            }
        } else if let Some(radar_distance) = track.radar_distance() {
            format!(
                "{} {:.1}m (Radar:{:.1}m)",
                track.class_name(),
                track.distance(),
                radar_distance
            )
        } else {
            format!("{} {:.1}m", track.class_name(), track.distance())
        };

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
            COLOR_GREEN_SCALAR,
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

        let center_x = track.image_coordinate_x() as i32;
        let center_y = track.image_coordinate_y() as i32;
        opencv::imgproc::circle(
            &mut visualization,
            Point::new(center_x, center_y),
            5,
            COLOR_GREEN_SCALAR,
            -1,
            LINE_8,
            0,
        )?;

        if let Some(lidar_detection) = &track.lidar_detection {
            let lidar_x = lidar_detection.image_coordinate_x as i32;
            let lidar_y = lidar_detection.image_coordinate_y as i32;
            opencv::imgproc::circle(
                &mut visualization,
                Point::new(lidar_x, lidar_y),
                2,
                COLOR_GREEN_SCALAR,
                -1,
                LINE_8,
                0,
            )?;
        }
    }

    let info_text = format!(
        "Fused: {} | Camera-only: {} | Radar-only: {} | Lidar-only: {}",
        fusion_result.fused_tracks.len(),
        fusion_result.unmatched_camera_detections.len(),
        fusion_result.unmatched_radar_detections.len(),
        fusion_result.unmatched_lidar_detections.len()
    );
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

    let legend_text = "Green Box = Fused Track | Blue Box = Camera Only | Radar/Lidar Point Color: Red (Near) -> Yellow (Mid) -> Green (Far)";
    put_text(
        &mut visualization,
        legend_text,
        Point::new(10, 60),
        HersheyFonts::FONT_HERSHEY_SIMPLEX as i32,
        0.3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        LINE_8,
        false,
    )?;

    log_rerun_image(
        recording,
        &visualization,
        FUSION_PROJECTION_CAM_FRONT_ENTITY_PATH,
    )?;

    Ok(())
}
