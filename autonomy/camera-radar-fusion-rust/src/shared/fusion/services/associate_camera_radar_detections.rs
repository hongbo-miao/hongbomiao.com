use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::utils::calculate_distance_to_bounding_box::calculate_distance_to_bounding_box;
use crate::shared::radar::types::radar_detection::RadarDetection;
use std::collections::HashSet;
use tracing::debug;

pub fn associate_camera_radar_detections(
    camera_detections: &[CameraDetection],
    radar_detections: &[RadarDetection],
    distance_threshold_pixels: f64,
) -> Vec<(usize, usize)> {
    if camera_detections.is_empty() || radar_detections.is_empty() {
        return Vec::new();
    }

    let mut distance_pairs: Vec<(f64, usize, usize)> = Vec::new();

    for (camera_index, camera_detection) in camera_detections.iter().enumerate() {
        for (radar_index, radar_detection) in radar_detections.iter().enumerate() {
            let distance = calculate_distance_to_bounding_box(radar_detection, camera_detection);

            if distance <= distance_threshold_pixels {
                distance_pairs.push((distance, camera_index, radar_index));
            }
        }
    }

    distance_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("Failed to compare distances"));

    let mut matched_pairs: Vec<(usize, usize)> = Vec::new();
    let mut used_camera_indices: HashSet<usize> = HashSet::new();
    let mut used_radar_indices: HashSet<usize> = HashSet::new();

    for (_distance, camera_index, radar_index) in distance_pairs {
        if !used_camera_indices.contains(&camera_index)
            && !used_radar_indices.contains(&radar_index)
        {
            matched_pairs.push((camera_index, radar_index));
            used_camera_indices.insert(camera_index);
            used_radar_indices.insert(radar_index);
        }
    }

    debug!(
        "Associated {} pairs from {} camera and {} radar detections",
        matched_pairs.len(),
        camera_detections.len(),
        radar_detections.len()
    );

    matched_pairs
}
