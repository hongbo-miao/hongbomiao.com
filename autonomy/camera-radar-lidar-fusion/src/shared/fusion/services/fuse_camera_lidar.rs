use crate::config::AppConfig;
use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::services::associate_camera_lidar_detections::associate_camera_lidar_detections;
use crate::shared::fusion::services::create_fused_track::create_fused_track;
use crate::shared::fusion::types::fused_track::FusedTrack;
use crate::shared::lidar::types::lidar_detection::LidarDetection;
use std::collections::HashSet;
use tracing::info;

pub struct FusionResult {
    pub fused_tracks: Vec<FusedTrack>,
    pub unmatched_camera_detections: Vec<CameraDetection>,
    pub unmatched_lidar_detections: Vec<LidarDetection>,
}

pub fn fuse_camera_lidar(
    camera_detections: Vec<CameraDetection>,
    lidar_detections: Vec<LidarDetection>,
    distance_threshold_pixels: f64,
    config: &AppConfig,
) -> FusionResult {
    let matched_pairs = associate_camera_lidar_detections(
        &camera_detections,
        &lidar_detections,
        distance_threshold_pixels,
    );

    let mut fused_tracks = Vec::new();
    let mut matched_camera_indices = HashSet::new();
    let mut matched_lidar_indices = HashSet::new();

    for (camera_index, lidar_index) in matched_pairs {
        let camera_detection = camera_detections[camera_index].clone();
        let lidar_detection = lidar_detections[lidar_index].clone();

        let track = create_fused_track(camera_detection, None, Some(lidar_detection), config);
        fused_tracks.push(track);

        matched_camera_indices.insert(camera_index);
        matched_lidar_indices.insert(lidar_index);
    }

    let unmatched_camera_detections: Vec<CameraDetection> = camera_detections
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !matched_camera_indices.contains(index))
        .map(|(_, detection)| detection)
        .collect();

    let unmatched_lidar_detections: Vec<LidarDetection> = lidar_detections
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !matched_lidar_indices.contains(index))
        .map(|(_, detection)| detection)
        .collect();

    info!(
        "Fusion results: {} fused tracks, {} camera-only, {} lidar-only",
        fused_tracks.len(),
        unmatched_camera_detections.len(),
        unmatched_lidar_detections.len()
    );

    FusionResult {
        fused_tracks,
        unmatched_camera_detections,
        unmatched_lidar_detections,
    }
}
