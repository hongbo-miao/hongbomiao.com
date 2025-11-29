#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod shared;

use crate::config::AppConfig;
use crate::shared::camera::services::detect_objects_in_camera::YoloModel;
use crate::shared::camera::services::log_camera_to_rerun::log_camera_to_rerun;
use crate::shared::camera::services::visualize_camera_only::visualize_camera_only;
use crate::shared::fusion::services::visualize_camera_lidar_fusion::visualize_camera_lidar_fusion;
use crate::shared::fusion::services::visualize_camera_radar_fusion::visualize_camera_radar_fusion;
use crate::shared::fusion::services::visualize_camera_radar_lidar_fusion::visualize_camera_radar_lidar_fusion;
use crate::shared::lidar::services::load_lidar_data::load_lidar_data;
use crate::shared::lidar::services::log_lidar_to_rerun::log_lidar_to_rerun;
use crate::shared::lidar::utils::transform_lidar_to_vehicle::transform_lidar_to_vehicle;
use crate::shared::map::services::load_ego_pose::load_ego_poses;
use crate::shared::map::services::log_ego_position_to_rerun::log_ego_position_to_rerun;
use crate::shared::map::services::log_ego_trajectory_to_rerun::log_ego_trajectory_to_rerun;
use crate::shared::map::services::log_ego_vehicle_to_rerun::log_ego_vehicle_to_rerun;
use crate::shared::nuscenes::types::nuscenes_calibrated_sensor::NuscenesCalibratedSensor;
use crate::shared::nuscenes::types::nuscenes_log::NuscenesLog;
use crate::shared::nuscenes::types::nuscenes_sample::NuscenesSample;
use crate::shared::nuscenes::types::nuscenes_sample_data::NuscenesSampleData;
use crate::shared::nuscenes::types::nuscenes_scene::NuscenesScene;
use crate::shared::nuscenes::types::nuscenes_sensor::NuscenesSensor;
use crate::shared::occupancy::services::build_occupancy_grid_from_lidar::build_occupancy_grid_from_lidar;
use crate::shared::occupancy::services::clear_distant_voxels::clear_distant_voxels;
use crate::shared::occupancy::services::decay_occupancy_probabilities::decay_occupancy_probabilities;
use crate::shared::occupancy::services::log_occupancy_grid_to_rerun::log_occupancy_grid_to_rerun;
use crate::shared::occupancy::types::occupancy_grid::{OccupancyGrid, OccupancyGridConfig};
use crate::shared::radar::services::load_radar_data::load_radar_data;
use crate::shared::radar::services::log_radar_to_rerun::log_radar_to_rerun;
use anyhow::{Context, Result, bail};
use nalgebra::{Matrix3, Matrix4, Quaternion, UnitQuaternion};
use rerun as rr;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

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

    // Initialize Rerun for 3D visualization
    let recording = rr::RecordingStreamBuilder::new("camera_radar_lidar_fusion")
        .spawn()
        .context("Failed to spawn Rerun viewer")?;

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

    let nuscenes_scenes: Vec<NuscenesScene> = read_json_array(&json_root.join("scene.json"))?;
    let nuscenes_samples: Vec<NuscenesSample> = read_json_array(&json_root.join("sample.json"))?;
    let nuscenes_sample_data: Vec<NuscenesSampleData> =
        read_json_array(&json_root.join("sample_data.json"))?;
    let nuscenes_calibrated_sensors: Vec<NuscenesCalibratedSensor> =
        read_json_array(&json_root.join("calibrated_sensor.json"))?;
    let nuscenes_sensors: Vec<NuscenesSensor> = read_json_array(&json_root.join("sensor.json"))?;
    let nuscenes_logs: Vec<NuscenesLog> = read_json_array(&json_root.join("log.json"))?;
    let ego_poses = load_ego_poses(&json_root)?;

    // Index by token for quick lookup
    let mut nuscenes_sample_by_token = HashMap::new();
    for nuscenes_sample in &nuscenes_samples {
        nuscenes_sample_by_token.insert(nuscenes_sample.token.clone(), nuscenes_sample);
    }

    let mut nuscenes_sample_data_by_sample: HashMap<&str, Vec<&NuscenesSampleData>> =
        HashMap::new();
    for nuscenes_sample_data_item in &nuscenes_sample_data {
        nuscenes_sample_data_by_sample
            .entry(nuscenes_sample_data_item.sample_token.as_str())
            .or_default()
            .push(nuscenes_sample_data_item);
    }

    let mut nuscenes_calibrated_sensor_by_token = HashMap::new();
    for nuscenes_calibrated_sensor in &nuscenes_calibrated_sensors {
        nuscenes_calibrated_sensor_by_token.insert(
            nuscenes_calibrated_sensor.token.clone(),
            nuscenes_calibrated_sensor,
        );
    }

    let mut nuscenes_sensor_channel_by_token: HashMap<&str, &str> = HashMap::new();
    for nuscenes_sensor in &nuscenes_sensors {
        nuscenes_sensor_channel_by_token.insert(
            nuscenes_sensor.token.as_str(),
            nuscenes_sensor.channel.as_str(),
        );
    }

    let mut channel_by_calibration: HashMap<&str, &str> = HashMap::new();
    for nuscenes_calibrated_sensor in &nuscenes_calibrated_sensors {
        if let Some(channel) =
            nuscenes_sensor_channel_by_token.get(nuscenes_calibrated_sensor.sensor_token.as_str())
        {
            channel_by_calibration.insert(nuscenes_calibrated_sensor.token.as_str(), *channel);
        }
    }

    // Build log lookup map
    let nuscenes_log_by_token: HashMap<_, _> = nuscenes_logs
        .iter()
        .map(|nuscenes_log| (nuscenes_log.token.clone(), nuscenes_log))
        .collect();

    // Scene index and frame cap
    let scene_index: usize = 0;
    let nuscenes_scene = &nuscenes_scenes[scene_index];
    tracing::info!("Processing scene {scene_index} ({})", nuscenes_scene.name);

    // Get location for GPS coordinate conversion
    let location = nuscenes_log_by_token
        .get(&nuscenes_scene.log_token)
        .map(|nuscenes_log| nuscenes_log.location.as_str())
        .unwrap_or("singapore-onenorth");

    // Prepare YOLO model
    let mut yolo_model = YoloModel::new(Path::new(&config.yolo_model_path))?;

    // Initialize occupancy grid from config
    let occupancy_grid_config = OccupancyGridConfig::new(
        config.occupancy_voxel_size_m,
        nalgebra::Vector3::new(
            config.occupancy_min_bound_x_m,
            config.occupancy_min_bound_y_m,
            config.occupancy_min_bound_z_m,
        ),
        nalgebra::Vector3::new(
            config.occupancy_max_bound_x_m,
            config.occupancy_max_bound_y_m,
            config.occupancy_max_bound_z_m,
        ),
        config.occupancy_occupied_threshold,
        config.occupancy_free_threshold,
        config.occupancy_occupied_probability_increment,
        config.occupancy_free_probability_decrement,
    );
    let mut occupancy_grid = OccupancyGrid::new(occupancy_grid_config);

    // Walk samples via next chain
    let mut current_token = nuscenes_scene.first_sample_token.clone();
    let mut frames = 0usize;
    let mut trajectory_ego_poses = Vec::new();

    while !current_token.is_empty() && frames < config.max_frame_count {
        let nuscenes_sample = match nuscenes_sample_by_token.get(&current_token) {
            Some(nuscenes_sample_value) => *nuscenes_sample_value,
            None => break,
        };
        let nuscenes_sample_data_list =
            match nuscenes_sample_data_by_sample.get(nuscenes_sample.token.as_str()) {
                Some(list) => list,
                None => {
                    current_token = nuscenes_sample.next.clone();
                    continue;
                }
            };

        let mut camera_sample_data_option: Option<&NuscenesSampleData> = None;
        let mut radar_sample_data_option: Option<&NuscenesSampleData> = None;
        let mut lidar_sample_data_option: Option<&NuscenesSampleData> = None;
        for nuscenes_sample_data_item in nuscenes_sample_data_list {
            let channel = channel_by_calibration
                .get(nuscenes_sample_data_item.calibrated_sensor_token.as_str())
                .copied()
                .unwrap_or_else(|| infer_channel(&nuscenes_sample_data_item.filename));
            let is_sample_path = nuscenes_sample_data_item.filename.contains("samples/");
            match channel {
                "CAM_FRONT" => {
                    if camera_sample_data_option.is_none()
                        || (nuscenes_sample_data_item.is_key_frame && is_sample_path)
                    {
                        camera_sample_data_option = Some(nuscenes_sample_data_item);
                    }
                }
                "RADAR_FRONT" => {
                    if radar_sample_data_option.is_none()
                        || (nuscenes_sample_data_item.is_key_frame && is_sample_path)
                    {
                        radar_sample_data_option = Some(nuscenes_sample_data_item);
                    }
                }
                "LIDAR_TOP" => {
                    if lidar_sample_data_option.is_none()
                        || (nuscenes_sample_data_item.is_key_frame && is_sample_path)
                    {
                        lidar_sample_data_option = Some(nuscenes_sample_data_item);
                    }
                }
                _ => {}
            }
        }

        // Camera is required (industry standard)
        let camera_sample_data = match camera_sample_data_option {
            Some(nuscenes_sample_data_item)
                if nuscenes_sample_data_item.filename.contains("CAM_FRONT") =>
            {
                nuscenes_sample_data_item
            }
            _ => {
                tracing::warn!(
                    "Sample {} has no CAM_FRONT data; skipping",
                    nuscenes_sample.token
                );
                current_token = nuscenes_sample.next.clone();
                continue;
            }
        };

        // Log and collect ego pose for trajectory
        if let Some(ego_pose) = ego_poses.get(&camera_sample_data.ego_pose_token) {
            // Log current position to MapView
            if let Err(error) = log_ego_position_to_rerun(
                &recording,
                ego_pose,
                location,
                "world/ego_vehicle/position",
            ) {
                tracing::warn!("Failed to log ego position: {error}");
            }

            // Collect for trajectory
            trajectory_ego_poses.push(ego_pose);

            // Log accumulated trajectory so far (growing line)
            if let Err(error) = log_ego_trajectory_to_rerun(
                &recording,
                &trajectory_ego_poses,
                location,
                "world/ego_vehicle/trajectory",
            ) {
                tracing::warn!("Failed to log ego trajectory: {error}");
            }
        }

        // Radar and Lidar are optional
        let radar_sample_data_result =
            radar_sample_data_option.filter(|nuscenes_sample_data_item| {
                nuscenes_sample_data_item.filename.contains("RADAR_FRONT")
            });
        let lidar_sample_data_result =
            lidar_sample_data_option.filter(|nuscenes_sample_data_item| {
                nuscenes_sample_data_item.filename.contains("LIDAR_TOP")
            });

        // Get camera calibration (required)
        let camera_calibration = match nuscenes_calibrated_sensor_by_token
            .get(&camera_sample_data.calibrated_sensor_token)
        {
            Some(nuscenes_calibrated_sensor_value) => *nuscenes_calibrated_sensor_value,
            None => {
                tracing::warn!(
                    "Missing calibration for camera sensor {}; skipping",
                    camera_sample_data.calibrated_sensor_token
                );
                current_token = nuscenes_sample.next.clone();
                continue;
            }
        };

        // Validate camera intrinsic
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
            current_token = nuscenes_sample.next.clone();
            continue;
        }
        let flat_intrinsic: Vec<f64> = camera_calibration
            .camera_intrinsic
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let camera_intrinsic = Matrix3::from_row_slice(&flat_intrinsic);

        // Get optional radar calibration
        let radar_calibration_result = radar_sample_data_result.and_then(|radar_sample_data| {
            nuscenes_calibrated_sensor_by_token
                .get(&radar_sample_data.calibrated_sensor_token)
                .copied()
        });

        // Get optional lidar calibration
        let lidar_calibration_result = lidar_sample_data_result.and_then(|lidar_sample_data| {
            nuscenes_calibrated_sensor_by_token
                .get(&lidar_sample_data.calibrated_sensor_token)
                .copied()
        });

        // Build camera transforms
        let camera_to_vehicle =
            build_transform(camera_calibration.rotation, camera_calibration.translation);
        let vehicle_to_camera = camera_to_vehicle
            .try_inverse()
            .context("Failed to invert camera_to_vehicle")?;

        // Camera image path (always available)
        let camera_image_path = files_root.join(Path::new(&camera_sample_data.filename));

        let radar_info = radar_sample_data_result.zip(radar_calibration_result).map(
            |(radar_sample_data, radar_calibration)| {
                let radar_to_vehicle =
                    build_transform(radar_calibration.rotation, radar_calibration.translation);
                let radar_to_camera = vehicle_to_camera * radar_to_vehicle;
                let radar_file_path = files_root.join(Path::new(&radar_sample_data.filename));
                (radar_to_camera, radar_file_path)
            },
        );

        let lidar_info = lidar_sample_data_result.zip(lidar_calibration_result).map(
            |(lidar_sample_data, lidar_calibration)| {
                let lidar_to_vehicle =
                    build_transform(lidar_calibration.rotation, lidar_calibration.translation);
                let lidar_to_camera = vehicle_to_camera * lidar_to_vehicle;
                let lidar_file_path = files_root.join(Path::new(&lidar_sample_data.filename));
                (lidar_to_camera, lidar_to_vehicle, lidar_file_path)
            },
        );

        // Log sensor data to Rerun for 3D visualization
        if let Err(error) =
            log_camera_to_rerun(&recording, &camera_image_path, "world/camera/image")
        {
            tracing::warn!("Failed to log camera image: {error}");
        }

        // Log ego vehicle representation (box + forward arrow)
        if let Err(error) = log_ego_vehicle_to_rerun(
            &recording,
            "world/ego_vehicle",
            config.ego_vehicle_half_length_m,
            config.ego_vehicle_half_width_m,
            config.ego_vehicle_half_height_m,
            config.ego_vehicle_elevation_m,
        ) {
            tracing::warn!("Failed to log ego vehicle: {error}");
        }

        if let Some((_, ref radar_file_path)) = radar_info
            && let Ok(radar_data) = load_radar_data(radar_file_path)
            && let Err(error) = log_radar_to_rerun(&recording, &radar_data, "world/sensors/radar")
        {
            tracing::warn!("Failed to log radar data: {error}");
        }
        if let Some((_, ref lidar_to_vehicle, ref lidar_file_path)) = lidar_info
            && let Ok(lidar_data) = load_lidar_data(lidar_file_path)
        {
            // Log raw LiDAR point cloud (sensor frame)
            if let Err(error) = log_lidar_to_rerun(&recording, &lidar_data, "world/sensors/lidar") {
                tracing::warn!("Failed to log lidar data: {error}");
            }

            // Apply decay to existing voxels (for dynamic obstacle handling)
            decay_occupancy_probabilities(&mut occupancy_grid, config.occupancy_decay_rate);

            // Clear voxels outside local area (keep grid centered on vehicle)
            clear_distant_voxels(&mut occupancy_grid, config.occupancy_clear_distance_m);

            // Transform LiDAR points to vehicle frame
            if let Ok(lidar_data_vehicle) =
                transform_lidar_to_vehicle(&lidar_data, lidar_to_vehicle)
            {
                // Sensor origin in vehicle frame (from lidar_to_vehicle translation)
                let sensor_origin = lidar_to_vehicle.column(3).xyz().cast::<f32>();

                // Build occupancy grid from LiDAR data in vehicle frame
                if let Err(error) = build_occupancy_grid_from_lidar(
                    &mut occupancy_grid,
                    &lidar_data_vehicle,
                    &sensor_origin,
                ) {
                    tracing::warn!("Failed to build occupancy grid: {error}");
                } else {
                    // Log occupancy grid visualization
                    if let Err(error) = log_occupancy_grid_to_rerun(
                        &recording,
                        &occupancy_grid,
                        "world/occupancy_grid",
                    ) {
                        tracing::warn!("Failed to log occupancy grid: {error}");
                    } else {
                        tracing::info!(
                            "Occupancy grid: {} occupied, {} free, {} unknown voxels (vehicle frame)",
                            occupancy_grid.occupied_voxel_count(),
                            occupancy_grid.free_voxel_count(),
                            occupancy_grid.unknown_voxel_count()
                        );
                    }
                }
            } else {
                tracing::warn!("Failed to transform LiDAR data to vehicle frame");
            }
        }

        // Route to appropriate visualization based on available sensors
        match (radar_info, lidar_info) {
            // Case 1: Camera + Radar + Lidar (all three sensors)
            (
                Some((radar_to_camera, radar_file_path)),
                Some((lidar_to_camera, _lidar_to_vehicle, lidar_file_path)),
            ) => {
                tracing::info!("Using Camera + Radar + Lidar fusion");
                visualize_camera_radar_lidar_fusion(
                    &recording,
                    &camera_image_path,
                    &radar_file_path,
                    &lidar_file_path,
                    camera_intrinsic,
                    radar_to_camera,
                    lidar_to_camera,
                    &mut yolo_model,
                    AppConfig::get().clone(),
                )?;
            }

            // Case 2: Camera + Radar (no lidar)
            (Some((radar_to_camera, radar_file_path)), None) => {
                tracing::info!("Using Camera + Radar fusion");
                visualize_camera_radar_fusion(
                    &recording,
                    &camera_image_path,
                    &radar_file_path,
                    camera_intrinsic,
                    radar_to_camera,
                    &mut yolo_model,
                    AppConfig::get().clone(),
                )?;
            }

            // Case 3: Camera + Lidar (no radar)
            (None, Some((lidar_to_camera, _lidar_to_vehicle, lidar_file_path))) => {
                tracing::info!("Using Camera + Lidar fusion");
                visualize_camera_lidar_fusion(
                    &recording,
                    &camera_image_path,
                    &lidar_file_path,
                    camera_intrinsic,
                    lidar_to_camera,
                    &mut yolo_model,
                    AppConfig::get().clone(),
                )?;
            }

            // Case 4: Camera only (no radar, no lidar)
            (None, None) => {
                tracing::info!("Using Camera only");
                visualize_camera_only(
                    &recording,
                    &camera_image_path,
                    &mut yolo_model,
                    AppConfig::get().clone(),
                )?;
            }
        }

        frames += 1;
        if current_token == nuscenes_sample.next {
            break;
        }
        current_token = nuscenes_sample.next.clone();
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
    } else if filename.contains("LIDAR_TOP") {
        "LIDAR_TOP"
    } else {
        "UNKNOWN"
    }
}
