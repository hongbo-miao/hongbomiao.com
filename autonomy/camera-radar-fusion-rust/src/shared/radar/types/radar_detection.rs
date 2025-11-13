use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarDetection {
    pub position_3d: Vector3<f64>,
    pub velocity: f64,
    pub radar_cross_section: f64,
    pub image_coordinate_x: f64,
    pub image_coordinate_y: f64,
    pub distance: f64,
}

impl RadarDetection {
    pub fn new(
        position_3d: Vector3<f64>,
        velocity: f64,
        radar_cross_section: f64,
        image_coordinate_x: f64,
        image_coordinate_y: f64,
    ) -> Self {
        let distance = position_3d.norm();
        Self {
            position_3d,
            velocity,
            radar_cross_section,
            image_coordinate_x,
            image_coordinate_y,
            distance,
        }
    }
}

impl std::fmt::Display for RadarDetection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "RadarDetection(distance={:.1}m, velocity={:.1}m/s, image_pos=({:.0}, {:.0}))",
            self.distance, self.velocity, self.image_coordinate_x, self.image_coordinate_y
        )
    }
}
