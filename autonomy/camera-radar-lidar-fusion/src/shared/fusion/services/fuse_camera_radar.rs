use crate::config::AppConfig;
use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::services::associate_camera_radar_detections::associate_camera_radar_detections;
use crate::shared::fusion::services::create_fused_track::create_fused_track;
use crate::shared::fusion::types::fused_track::FusedTrack;
use crate::shared::radar::types::radar_detection::RadarDetection;
use std::collections::HashSet;
use tracing::info;

pub struct FusionResult {
    pub fused_tracks: Vec<FusedTrack>,
    pub unmatched_camera_detections: Vec<CameraDetection>,
    pub unmatched_radar_detections: Vec<RadarDetection>,
}

pub fn fuse_camera_radar(
    camera_detections: Vec<CameraDetection>,
    radar_detections: Vec<RadarDetection>,
    distance_threshold_pixels: f64,
    config: &AppConfig,
) -> FusionResult {
    let matched_pairs = associate_camera_radar_detections(
        &camera_detections,
        &radar_detections,
        distance_threshold_pixels,
    );

    let mut fused_tracks = Vec::new();
    let mut matched_camera_indices = HashSet::new();
    let mut matched_radar_indices = HashSet::new();

    for (camera_index, radar_index) in matched_pairs {
        let camera_detection = camera_detections[camera_index].clone();
        let radar_detection = radar_detections[radar_index].clone();
        let track = create_fused_track(camera_detection, Some(radar_detection), None, config);
        fused_tracks.push(track);

        matched_camera_indices.insert(camera_index);
        matched_radar_indices.insert(radar_index);
    }

    let unmatched_camera_detections: Vec<CameraDetection> = camera_detections
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !matched_camera_indices.contains(index))
        .map(|(_, detection)| detection)
        .collect();

    let unmatched_radar_detections: Vec<RadarDetection> = radar_detections
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !matched_radar_indices.contains(index))
        .map(|(_, detection)| detection)
        .collect();

    info!(
        "Fusion results: {} fused tracks, {} camera-only, {} radar-only",
        fused_tracks.len(),
        unmatched_camera_detections.len(),
        unmatched_radar_detections.len()
    );

    FusionResult {
        fused_tracks,
        unmatched_camera_detections,
        unmatched_radar_detections,
    }
}
