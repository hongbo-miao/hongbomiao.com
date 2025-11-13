use crate::config::AppConfig;
use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::radar::types::radar_detection::RadarDetection;
use nalgebra::Vector4;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedTrack {
    pub camera_detection: CameraDetection,
    pub radar_detection: RadarDetection,
    pub fusion_confidence: f64,
}

impl FusedTrack {
    pub fn new(
        camera_detection: CameraDetection,
        radar_detection: RadarDetection,
        fusion_confidence: f64,
    ) -> Self {
        Self {
            camera_detection,
            radar_detection,
            fusion_confidence,
        }
    }

    pub fn bounding_box(&self) -> Vector4<f64> {
        self.camera_detection.bounding_box
    }

    pub fn class_name(&self) -> &str {
        &self.camera_detection.class_name
    }

    pub fn velocity(&self) -> f64 {
        self.radar_detection.velocity
    }

    pub fn distance(&self) -> f64 {
        self.radar_detection.distance
    }

    pub fn image_coordinate_x(&self) -> f64 {
        self.camera_detection.center_x()
    }

    pub fn image_coordinate_y(&self) -> f64 {
        self.camera_detection.center_y()
    }

    pub fn is_moving(&self, config: &AppConfig) -> bool {
        self.velocity() > config.movement_velocity_threshold_mps
    }
}

impl std::fmt::Display for FusedTrack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FusedTrack(class={}, distance={:.1}m, velocity={:.1}m/s, confidence={:.2})",
            self.class_name(),
            self.distance(),
            self.velocity(),
            self.fusion_confidence
        )
    }
}
