mod config;
mod shared;

use crate::config::AppConfig;
use crate::shared::camera::services::detect_objects_in_camera::YoloModel;
use crate::shared::fusion::services::visualize_camera_radar_fusion::visualize_camera_radar_fusion;
use anyhow::{Context, Result, bail};
use nalgebra::{Matrix3, Matrix4, Quaternion, UnitQuaternion};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Scene {
    pub first_sample_token: String,
    pub name: String,
}

#[derive(Debug, Deserialize)]
struct Sample {
    pub token: String,
    pub next: String,
    #[allow(dead_code)]
    pub prev: String,
    #[allow(dead_code)]
    pub scene_token: String,
}

#[derive(Debug, Deserialize)]
struct SampleData {
    #[allow(dead_code)]
    pub token: String,
    pub sample_token: String,
    pub calibrated_sensor_token: String,
    pub filename: String,
    pub is_key_frame: bool,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
}

#[derive(Debug, Deserialize)]
struct CalibratedSensor {
    pub token: String,
    pub sensor_token: String,
    pub rotation: [f64; 4],
    pub translation: [f64; 3],
    #[serde(default)]
    pub camera_intrinsic: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct Sensor {
    pub token: String,
    pub channel: String,
}

fn read_json_array<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let bytes = fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let items: Vec<T> = serde_json::from_slice(&bytes)
        .with_context(|| format!("Failed to parse JSON {}", path.display()))?;
    Ok(items)
}

fn build_transform(rotation_wxyz: [f64; 4], translation_xyz: [f64; 3]) -> Matrix4<f64> {
    let quaternion = Quaternion::new(
        rotation_wxyz[0],
        rotation_wxyz[1],
        rotation_wxyz[2],
        rotation_wxyz[3],
    );
    let unit_quaternion: UnitQuaternion<f64> = UnitQuaternion::from_quaternion(quaternion);
    let rotation_matrix = unit_quaternion.to_rotation_matrix();
    let mut transform_matrix = Matrix4::<f64>::identity();
    transform_matrix
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(rotation_matrix.matrix());
    transform_matrix[(0, 3)] = translation_xyz[0];
    transform_matrix[(1, 3)] = translation_xyz[1];
    transform_matrix[(2, 3)] = translation_xyz[2];
    transform_matrix
}

fn run_visualization() -> Result<()> {
    let config = AppConfig::get();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let dataset_root = Path::new("data/v1.0-mini");
    let json_root = if dataset_root.join("scene.json").exists() {
        dataset_root.to_path_buf()
    } else if dataset_root.join("v1.0-mini/scene.json").exists() {
        dataset_root.join("v1.0-mini")
    } else {
        bail!(
            "Could not locate scene.json under {} or {}/v1.0-mini",
            dataset_root.display(),
            dataset_root.display()
        );
    };

    let files_root = if dataset_root.join("samples").exists() {
        dataset_root.to_path_buf()
    } else {
        json_root
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| json_root.clone())
    };

    let scenes: Vec<Scene> = read_json_array(&json_root.join("scene.json"))?;
    let samples: Vec<Sample> = read_json_array(&json_root.join("sample.json"))?;
    let sample_data: Vec<SampleData> = read_json_array(&json_root.join("sample_data.json"))?;
    let calibrated: Vec<CalibratedSensor> =
        read_json_array(&json_root.join("calibrated_sensor.json"))?;
    let sensors: Vec<Sensor> = read_json_array(&json_root.join("sensor.json"))?;

    // Index by token for quick lookup
    let mut sample_by_token = HashMap::new();
    for sample in &samples {
        sample_by_token.insert(sample.token.clone(), sample);
    }

    let mut sample_data_by_sample: HashMap<&str, Vec<&SampleData>> = HashMap::new();
    for sample_data_item in &sample_data {
        sample_data_by_sample
            .entry(sample_data_item.sample_token.as_str())
            .or_default()
            .push(sample_data_item);
    }

    let mut calib_by_token = HashMap::new();
    for calibration in &calibrated {
        calib_by_token.insert(calibration.token.clone(), calibration);
    }

    let mut sensor_channel_by_token: HashMap<&str, &str> = HashMap::new();
    for sensor in &sensors {
        sensor_channel_by_token.insert(sensor.token.as_str(), sensor.channel.as_str());
    }

    let mut channel_by_calibration: HashMap<&str, &str> = HashMap::new();
    for calibration in &calibrated {
        if let Some(channel) = sensor_channel_by_token.get(calibration.sensor_token.as_str()) {
            channel_by_calibration.insert(calibration.token.as_str(), *channel);
        }
    }

    // Scene index and frame cap
    let scene_index: usize = 0;
    let scene = &scenes[scene_index];
    tracing::info!("Processing scene {scene_index} ({})", scene.name);

    // Prepare YOLO model
    let mut yolo_model = YoloModel::new(Path::new(&config.yolo_model_path))?;

    // Walk samples via next chain
    let mut current_token = scene.first_sample_token.clone();
    let mut frames = 0usize;

    while !current_token.is_empty() && frames < config.max_frame_count {
        let sample = match sample_by_token.get(&current_token) {
            Some(sample_value) => *sample_value,
            None => break,
        };
        let sample_datas = match sample_data_by_sample.get(sample.token.as_str()) {
            Some(list) => list,
            None => {
                current_token = sample.next.clone();
                continue;
            }
        };

        let mut camera_sample_data_option: Option<&SampleData> = None;
        let mut radar_sample_data_option: Option<&SampleData> = None;
        for sample_data_item in sample_datas {
            let channel = channel_by_calibration
                .get(sample_data_item.calibrated_sensor_token.as_str())
                .copied()
                .unwrap_or_else(|| infer_channel(&sample_data_item.filename));
            let is_sample_path = sample_data_item.filename.contains("samples/");
            match channel {
                "CAM_FRONT" => {
                    if camera_sample_data_option.is_none()
                        || (sample_data_item.is_key_frame && is_sample_path)
                    {
                        camera_sample_data_option = Some(sample_data_item);
                    }
                }
                "RADAR_FRONT" => {
                    if radar_sample_data_option.is_none()
                        || (sample_data_item.is_key_frame && is_sample_path)
                    {
                        radar_sample_data_option = Some(sample_data_item);
                    }
                }
                _ => {}
            }
        }

        let camera_sample_data = match camera_sample_data_option {
            Some(sample_data_item) if sample_data_item.filename.contains("CAM_FRONT") => {
                sample_data_item
            }
            _ => {
                tracing::warn!("Sample {} has no CAM_FRONT data", sample.token);
                current_token = sample.next.clone();
                continue;
            }
        };
        let radar_sample_data = match radar_sample_data_option {
            Some(sample_data_item) if sample_data_item.filename.contains("RADAR_FRONT") => {
                sample_data_item
            }
            _ => {
                tracing::warn!("Sample {} has no RADAR_FRONT data", sample.token);
                current_token = sample.next.clone();
                continue;
            }
        };

        let camera_calibration =
            match calib_by_token.get(&camera_sample_data.calibrated_sensor_token) {
                Some(calibration_value) => *calibration_value,
                None => {
                    tracing::warn!(
                        "Missing calibration for camera sensor {}",
                        camera_sample_data.calibrated_sensor_token
                    );
                    current_token = sample.next.clone();
                    continue;
                }
            };
        let radar_calibration = match calib_by_token.get(&radar_sample_data.calibrated_sensor_token)
        {
            Some(calibration_value) => *calibration_value,
            None => {
                tracing::warn!(
                    "Missing calibration for radar sensor {}",
                    radar_sample_data.calibrated_sensor_token
                );
                current_token = sample.next.clone();
                continue;
            }
        };

        // Intrinsic
        if camera_calibration.camera_intrinsic.len() != 3
            || camera_calibration
                .camera_intrinsic
                .iter()
                .any(|row| row.len() != 3)
        {
            tracing::warn!(
                "Camera intrinsic not available for sensor {}; skipping frame",
                camera_sample_data.calibrated_sensor_token
            );
            current_token = sample.next.clone();
            continue;
        }
        let flat_intrinsic: Vec<f64> = camera_calibration
            .camera_intrinsic
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let camera_intrinsic = Matrix3::from_row_slice(&flat_intrinsic);

        // Transforms
        let radar_to_vehicle =
            build_transform(radar_calibration.rotation, radar_calibration.translation);
        let camera_to_vehicle =
            build_transform(camera_calibration.rotation, camera_calibration.translation);
        let vehicle_to_camera = camera_to_vehicle
            .try_inverse()
            .context("Failed to invert camera_to_vehicle")?;
        let radar_to_camera = vehicle_to_camera * radar_to_vehicle;

        // Absolute file paths
        let camera_image_path = files_root.join(Path::new(&camera_sample_data.filename));
        let radar_file_path = files_root.join(Path::new(&radar_sample_data.filename));

        // Use existing visualizer (it loads radar internally via load_radar_data)
        let should_continue = visualize_camera_radar_fusion(
            camera_image_path,
            radar_file_path,
            camera_intrinsic,
            radar_to_camera,
            &mut yolo_model,
            AppConfig::get().clone(),
        )?;

        if !should_continue {
            tracing::info!("Exiting visualization loop");
            break;
        }

        frames += 1;
        if current_token == sample.next {
            break;
        }
        current_token = sample.next.clone();
    }

    Ok(())
}

fn main() -> Result<()> {
    run_visualization()
}

fn infer_channel(filename: &str) -> &str {
    if filename.contains("CAM_FRONT") {
        "CAM_FRONT"
    } else if filename.contains("RADAR_FRONT") {
        "RADAR_FRONT"
    } else {
        "UNKNOWN"
    }
}
