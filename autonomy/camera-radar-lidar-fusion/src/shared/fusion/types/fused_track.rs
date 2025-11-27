use crate::config::AppConfig;
use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::lidar::types::lidar_detection::LidarDetection;
use crate::shared::radar::types::radar_detection::RadarDetection;
use nalgebra::Vector4;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedTrack {
    pub camera_detection: CameraDetection,
    pub radar_detection: Option<RadarDetection>,
    pub lidar_detection: Option<LidarDetection>,
    pub fusion_confidence: f64,
    pub fused_distance: f64,
}

impl FusedTrack {
    pub fn new(
        camera_detection: CameraDetection,
        radar_detection: Option<RadarDetection>,
        lidar_detection: Option<LidarDetection>,
        fusion_confidence: f64,
        fused_distance: f64,
    ) -> Self {
        Self {
            camera_detection,
            radar_detection,
            lidar_detection,
            fusion_confidence,
            fused_distance,
        }
    }

    pub fn bounding_box(&self) -> Vector4<f64> {
        self.camera_detection.bounding_box
    }

    pub fn class_name(&self) -> &str {
        &self.camera_detection.class_name
    }

    pub fn velocity(&self) -> f64 {
        match &self.radar_detection {
            Some(radar_detection) => radar_detection.velocity,
            None => 0.0,
        }
    }

    pub fn distance(&self) -> f64 {
        self.fused_distance
    }

    #[allow(dead_code)]
    pub fn radar_distance(&self) -> Option<f64> {
        self.radar_detection
            .as_ref()
            .map(|radar_detection| radar_detection.distance)
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

    pub fn lidar_distance(&self) -> Option<f64> {
        self.lidar_detection.as_ref().map(|lidar| lidar.distance)
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
