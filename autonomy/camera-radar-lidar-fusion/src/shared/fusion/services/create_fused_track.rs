use crate::config::AppConfig;
use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::types::fused_track::FusedTrack;
use crate::shared::fusion::utils::calculate_fused_distance::calculate_fused_distance;
use crate::shared::lidar::types::lidar_detection::LidarDetection;
use crate::shared::radar::types::radar_detection::RadarDetection;
use tracing::debug;

pub fn create_fused_track(
    camera_detection: CameraDetection,
    radar_detection: Option<RadarDetection>,
    lidar_detection: Option<LidarDetection>,
    config: &AppConfig,
) -> FusedTrack {
    let fusion_confidence = config.camera_confidence_weight * camera_detection.confidence
        + config.fusion_base_confidence;

    // Calculate fused distance using variance-weighted averaging
    let fused_distance = match (radar_detection.as_ref(), lidar_detection.as_ref()) {
        (Some(radar_detection), Some(lidar_detection)) => {
            let fused =
                calculate_fused_distance(lidar_detection.distance, radar_detection.distance);
            tracing::info!(
                "FUSION: lidar={:.2}m, radar={:.2}m, fused={:.2}m",
                lidar_detection.distance,
                radar_detection.distance,
                fused
            );
            fused
        }
        (Some(radar_detection), None) => {
            // Only radar available: use radar distance
            radar_detection.distance
        }
        (None, Some(lidar_detection)) => {
            // Only lidar available: use lidar distance
            lidar_detection.distance
        }
        (None, None) => {
            // Fallback when no distance source is available
            0.0
        }
    };

    let track = FusedTrack::new(
        camera_detection,
        radar_detection,
        lidar_detection,
        fusion_confidence,
        fused_distance,
    );

    debug!("Created fused track: {track}");
    track
}
