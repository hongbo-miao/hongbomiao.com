use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::utils::calculate_distance_lidar_to_bounding_box::calculate_distance_lidar_to_bounding_box;
use crate::shared::lidar::types::lidar_detection::LidarDetection;
use std::collections::HashSet;
use tracing::debug;

pub fn associate_camera_lidar_detections(
    camera_detections: &[CameraDetection],
    lidar_detections: &[LidarDetection],
    distance_threshold_pixels: f64,
) -> Vec<(usize, usize)> {
    if camera_detections.is_empty() || lidar_detections.is_empty() {
        return Vec::new();
    }

    let mut distance_pairs: Vec<(f64, usize, usize)> = Vec::new();

    for (camera_index, camera_detection) in camera_detections.iter().enumerate() {
        for (lidar_index, lidar_detection) in lidar_detections.iter().enumerate() {
            let distance =
                calculate_distance_lidar_to_bounding_box(lidar_detection, camera_detection);

            if distance <= distance_threshold_pixels {
                distance_pairs.push((distance, camera_index, lidar_index));
            }
        }
    }

    distance_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("Failed to compare distances"));

    let mut matched_pairs: Vec<(usize, usize)> = Vec::new();
    let mut used_camera_indices: HashSet<usize> = HashSet::new();
    let mut used_lidar_indices: HashSet<usize> = HashSet::new();

    for (_distance, camera_index, lidar_index) in distance_pairs {
        if !used_camera_indices.contains(&camera_index)
            && !used_lidar_indices.contains(&lidar_index)
        {
            matched_pairs.push((camera_index, lidar_index));
            used_camera_indices.insert(camera_index);
            used_lidar_indices.insert(lidar_index);
        }
    }

    debug!(
        "Associated {} pairs from {} camera and {} lidar detections",
        matched_pairs.len(),
        camera_detections.len(),
        lidar_detections.len()
    );

    matched_pairs
}
