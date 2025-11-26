use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LidarDetection {
    pub position_3d: Vector3<f64>,
    pub intensity: f64,
    pub image_coordinate_x: f64,
    pub image_coordinate_y: f64,
    pub distance: f64,
}

impl LidarDetection {
    pub fn new(
        position_3d: Vector3<f64>,
        intensity: f64,
        image_coordinate_x: f64,
        image_coordinate_y: f64,
    ) -> Self {
        let distance = position_3d.norm();
        Self {
            position_3d,
            intensity,
            image_coordinate_x,
            image_coordinate_y,
            distance,
        }
    }
}

impl std::fmt::Display for LidarDetection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "LidarDetection(distance={:.1}m, intensity={:.2}, image_pos=({:.0}, {:.0}))",
            self.distance, self.intensity, self.image_coordinate_x, self.image_coordinate_y
        )
    }
}
