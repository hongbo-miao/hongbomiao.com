use crate::config::AppConfig;
use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::services::associate_camera_lidar_detections::associate_camera_lidar_detections;
use crate::shared::fusion::services::associate_camera_radar_detections::associate_camera_radar_detections;
use crate::shared::fusion::services::create_fused_track::create_fused_track;
use crate::shared::fusion::types::fused_track::FusedTrack;
use crate::shared::lidar::types::lidar_detection::LidarDetection;
use crate::shared::radar::types::radar_detection::RadarDetection;
use std::collections::HashSet;
use tracing::info;

pub struct FusionResult {
    pub fused_tracks: Vec<FusedTrack>,
    pub unmatched_camera_detections: Vec<CameraDetection>,
    pub unmatched_radar_detections: Vec<RadarDetection>,
    pub unmatched_lidar_detections: Vec<LidarDetection>,
}

pub fn fuse_camera_radar_lidar(
    camera_detections: Vec<CameraDetection>,
    radar_detections: Vec<RadarDetection>,
    lidar_detections: Vec<LidarDetection>,
    distance_threshold_pixels: f64,
    config: &AppConfig,
) -> FusionResult {
    let camera_radar_pairs = associate_camera_radar_detections(
        &camera_detections,
        &radar_detections,
        distance_threshold_pixels,
    );

    let mut fused_tracks = Vec::new();
    let mut matched_camera_indices = HashSet::new();
    let mut matched_radar_indices = HashSet::new();

    for (camera_index, radar_index) in camera_radar_pairs {
        let camera_detection = camera_detections[camera_index].clone();
        let radar_detection = radar_detections[radar_index].clone();
        let track = create_fused_track(camera_detection, Some(radar_detection), None, config);
        fused_tracks.push(track);

        matched_camera_indices.insert(camera_index);
        matched_radar_indices.insert(radar_index);
    }

    let camera_lidar_pairs = associate_camera_lidar_detections(
        &camera_detections,
        &lidar_detections,
        distance_threshold_pixels,
    );

    let mut matched_lidar_indices = HashSet::new();

    for (camera_index, lidar_index) in camera_lidar_pairs {
        let lidar_detection = lidar_detections[lidar_index].clone();

        if matched_camera_indices.contains(&camera_index) {
            if let Some(track) = fused_tracks.iter_mut().find(|track| {
                track.camera_detection.bounding_box == camera_detections[camera_index].bounding_box
            }) {
                // Add lidar detection to the track
                track.lidar_detection = Some(lidar_detection.clone());

                if let Some(radar_detection) = &track.radar_detection {
                    // Recalculate fused distance with both radar and lidar
                    let fused = crate::shared::fusion::utils::calculate_fused_distance::calculate_fused_distance(
                        lidar_detection.distance,
                        radar_detection.distance,
                    );
                    track.fused_distance = fused;

                    info!(
                        "FUSION UPDATE: lidar={:.2}m, radar={:.2}m, fused={:.2}m",
                        lidar_detection.distance, radar_detection.distance, fused
                    );
                } else {
                    track.fused_distance = lidar_detection.distance;

                    info!(
                        "FUSION UPDATE (lidar only): lidar={:.2}m",
                        lidar_detection.distance
                    );
                }

                matched_lidar_indices.insert(lidar_index);
            }
        } else {
            // Create a new fused track for a camera-lidar match without an associated radar detection
            let camera_detection = camera_detections[camera_index].clone();
            let lidar_detection = lidar_detections[lidar_index].clone();

            let track = create_fused_track(camera_detection, None, Some(lidar_detection), config);
            fused_tracks.push(track);

            matched_camera_indices.insert(camera_index);
            matched_lidar_indices.insert(lidar_index);
        }
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

    let unmatched_lidar_detections: Vec<LidarDetection> = lidar_detections
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !matched_lidar_indices.contains(index))
        .map(|(_, detection)| detection)
        .collect();

    info!(
        "Fusion results: {} fused tracks, {} camera-only, {} radar-only, {} lidar-only",
        fused_tracks.len(),
        unmatched_camera_detections.len(),
        unmatched_radar_detections.len(),
        unmatched_lidar_detections.len()
    );

    FusionResult {
        fused_tracks,
        unmatched_camera_detections,
        unmatched_radar_detections,
        unmatched_lidar_detections,
    }
}
