use crate::config::AppConfig;
use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::types::fused_track::FusedTrack;
use crate::shared::radar::types::radar_detection::RadarDetection;
use tracing::debug;

pub fn create_fused_track(
    camera_detection: CameraDetection,
    radar_detection: RadarDetection,
    config: &AppConfig,
) -> FusedTrack {
    let fusion_confidence = config.camera_confidence_weight * camera_detection.confidence
        + config.fusion_base_confidence;

    let track = FusedTrack::new(camera_detection, radar_detection, fusion_confidence);

    debug!("Created fused track: {track}");
    track
}
